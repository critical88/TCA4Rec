from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os
from torch.optim import AdamW
import json
from tqdm import tqdm
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min
import numpy as np
from pytorch_lightning import LightningModule
from collections import defaultdict
import torch.distributed as dist
from utils.metrics import *

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.token_item_pairs = ([], [])  # 预先计算好的(all_tokens, all_item_ids)元组

class Trie:
    def __init__(self):
        self.root = TrieNode()
        # 预缓存字典，key为序列tuple，value为(node, is_end)
        self.path_cache = {}

    def insert(self, word, item_id=None):
        node = self.root
        path = []
        # 记录路径上的所有前缀
        for token in word:
            path.append(token)
            # 在父节点中记录当前token对应的item_id
            if item_id is not None:
                # 直接更新token_item_pairs
                all_tokens, all_item_ids = node.token_item_pairs
                # 检查是否已存在相同的(token, item_id)对
                # pair_exists = False
                # for i, (t, id_) in enumerate(zip(all_tokens, all_item_ids)):
                #     if t == token and id_ == item_id:
                #         pair_exists = True
                #         break
                all_tokens.append(token)
                all_item_ids.append(item_id)
            
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
            # 缓存当前路径及其节点
            self.path_cache[tuple(path)] = (node, False)
        node.is_end_of_word = True
        # 更新最终路径的结束状态
        self.path_cache[tuple(path)] = (node, True)

    def get_node_for_sequence(self, batch_idx, tokens):
        """从预缓存中获取序列对应的节点"""
        if not tokens.numel():  # 空序列
            return self.root, True
            
        # 将tokens转换为tuple以用作字典key
        token_tuple = tuple(token.item() for token in tokens)
        
        # 直接从预缓存中获取节点和状态
        if token_tuple in self.path_cache:
            return self.path_cache[token_tuple][0], True
            
        return None, False
        
    def get_all_item_ids_for_prefix(self, prefix):
        """获取给定前缀对应节点下的所有item_ids
        Args:
            prefix: token ID列表，表示前缀
        Returns:
            list: 该前缀下的所有item_id列表
        """
        # 遍历前缀找到对应节点
        node = self.root
        for token in prefix:
            if token not in node.children:
                return []  # 前缀不存在，返回空列表
            node = node.children[token]
        
        # 直接从节点的token_item_pairs中获取所有不重复的item_ids
        _, all_item_ids = node.token_item_pairs
        return list(set(all_item_ids))
        
    def get_token_item_pairs_for_prefix(self, prefix):
        """获取给定前缀对应节点的token_item_pairs
        Args:
            prefix: token ID列表，表示前缀
        Returns:
            tuple: (all_tokens, all_item_ids)元组
        """
        # 如果是空前缀，返回根节点的元组
        if not prefix:
            return self.root.token_item_pairs
            
        # 遍历前缀找到对应节点
        node = self.root
        for token in prefix:
            if token not in node.children:
                return ([], [])  # 前缀不存在，返回空元组
            node = node.children[token]
            
        # 返回该节点的token_item_pairs
        return node.token_item_pairs

    def get_all_next_tokens(self, node):
        """获取该节点的所有可能的下一个token"""
        return list(node.children.keys()) if node else []

    def search(self, word):
        node = self.root
        for token in word:
            if token not in node.children:
                return False
            node = node.children[token]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for token in prefix:
            if token not in node.children:
                return False
            node = node.children[token]
        return True

class LLM4Rec(LightningModule):
    def __init__(
        self,
        args,
        model_name_or_path: str = 'albert-base-v2',
        tokenizer: AutoTokenizer = None,
        num_labels: int = 2,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-6,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        n_user: int = 0,
        n_item: int = 0,
        datamodule=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="datamodule")
        self.model_path = args.model_path
        
        self.model_name = model_name_or_path
        self.args = args
        self.padding_item_id = 0
        self.item_title2iid = datamodule.item_df.set_index("title")['iid'].to_dict()
        self.item_dict = datamodule.item_df.set_index("iid")['title'].to_dict()
        
        self.valid_Ks = [1, 3]
        self.test_Ks = [1,3,5,10]


        self.n_user = n_user
        self.n_item = n_item

        self.tau = args.tau

        ### llm model
        self.model = AutoModelForCausalLM.from_pretrained (
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)})

        self.tokenizer = tokenizer 
        self.vocab_size = self.model.config.vocab_size
        self.model.return_dict = True
        self.model.resize_token_embeddings(len(self.tokenizer))  
        
        self.model.config.return_dict = (
                True
            )
        
        self.is_evaluation = not not args.ckpt_path
        self.model.config.use_cache = False

        

        self.alpha = args.alpha
        self.init_cf_model()
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)

        if self.args.use_feature:
            hidden_size = 64
            self.user_token_id = self.tokenizer("[UserEmb]", add_special_tokens=False)['input_ids'][0]
            self.item_token_id = self.tokenizer("[ItemEmb]", add_special_tokens=False)['input_ids'][0]
            self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 5), 
                                 nn.ReLU(), 
                                 nn.Linear(hidden_size * 5, self.model.config.hidden_size))

        self.val_info = []
        
        # Initialize the prefix tree
        self.trie = Trie()
        eos_token_id = self.tokenizer.eos_token_id
        for item_id, token_ids in datamodule.item_token_ids.items():
            self.trie.insert(token_ids + [eos_token_id], item_id)
        
        
        print("********************")


    
    def init_cf_model(self):
       
        if self.args.cf_model in ["sasrec"]:
            from models.backbone.SASRec import SASRec

            ### base_model
            cf_model_hidden_unit = 64
            cf_model_state_dict_path = f"cf_model/{self.args.cf_model}/{self.args.dataset}.pt"

            ### use SASRec as basemodel
            self.cf_model = SASRec(self.n_user, self.n_item, device="cpu", maxlen=10, hidden_units=cf_model_hidden_unit)
            self.cf_model.load_state_dict(torch.load(cf_model_state_dict_path))
            self.cf_model.eval()
        
        else:
            raise ValueError("not supported base model, please choose [sasrec]")

        for name, param in self.cf_model.named_parameters():
            param.requires_grad = False



    def forward(self, batch_idx, title_idx, user_idx, input_ids, hist_id, attention_mask, is_testing=False):
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        batch_index = torch.arange(batch_size).to(device)
        
        labels = input_ids.clone()
        labels[input_ids >= self.vocab_size] = -100

        for i in range(batch_size):
            idx = torch.where(input_ids[i] == self.tokenizer.bos_token_id)[0][-1]
            labels[i][:idx + 1] = -100
            labels[i][-1] = -100
        shift_token_cf_logits = None
        inputs_embs = self.wrap_embs(input_ids, user_idx, hist_id)
        decoder_outputs = self.model(inputs_embeds=inputs_embs, attention_mask=attention_mask)
        logits = decoder_outputs.logits

        user_embs = self.cf_model.get_user_embs(user_idx, hist_id).to(inputs_embs.dtype)
        item_embs = self.cf_model.get_item_embs(torch.arange(self.n_item + 1).to(device)).to(inputs_embs.dtype)
        cf_logits = torch.matmul(user_embs, item_embs.t())

        allowed_tokens_mask = torch.zeros_like(logits, dtype=torch.bool)
        # 创建一个存储token的CF logits分布的张量
        token_cf_logits = torch.zeros_like(logits)
            
        for batch_idx in range(batch_size):
            # Get the start of generated tokens (after the last BOS token)
            generated_start_idx = (input_ids[batch_idx] == self.tokenizer.bos_token_id).nonzero(as_tuple=True)[0][-1] + 1
            generated_tokens = input_ids[batch_idx, generated_start_idx:]
            # 使用Trie类的缓存机制获取节点
            # node, valid_path = self.trie.get_node_for_sequence(batch_idx, generated_tokens)
            for t in range(len(generated_tokens)):
                node, valid_path = self.trie.get_node_for_sequence(batch_idx, generated_tokens[:t])
                if node.is_end_of_word or len(node.children) == 0:
                    continue
                else:
                    # 设置允许的token为True
                    next_tokens = list(node.children.keys())
                    allowed_tokens_mask[batch_idx, generated_start_idx + t - 1, next_tokens] = True
                    
                    # 使用scatter_sum高效计算每个token的CF logits分布
                    # 直接从节点中获取预先计算好的token_item_pairs
                    all_tokens, all_item_ids = node.token_item_pairs
                    
                    # # 过滤出允许的tokens
                    # next_tokens_set = set(next_tokens)
                    # filtered_indices = [i for i, token in enumerate(all_tokens) if token in next_tokens_set]
                    
                    # if filtered_indices:  # 如果有允许的tokens
                    #     all_tokens = [all_tokens[i] for i in filtered_indices]
                    #     all_item_ids = [all_item_ids[i] for i in filtered_indices]
                    
                    if all_tokens:  # 如果有数据要处理
                        # 将列表转换为张量
                        all_tokens = torch.tensor(all_tokens, device=device)
                        all_item_ids = torch.tensor(all_item_ids, device=device)
                        
                        # 使用scatter_sum计算每个token的CF logits总和
                        # 创建空的目标张量，大小为词表大小
                        beta = 1
                        all_cf_logits = cf_logits[batch_idx][all_item_ids]
                        # filtered_idx = (all_cf_logits > 0) | (all_item_ids == title_idx[batch_idx])
                        # filtered_cf_logits = all_cf_logits[filtered_idx]
                        # filtered_tokens = all_tokens[filtered_idx]
                        # token_sums = scatter_sum((filtered_cf_logits / beta).softmax(dim=-1) , filtered_tokens, dim=-1, dim_size=logits.size(-1))
                        # if all_cf_logits.shape[-1] < 10:
                        #     top10_v, top10_idx = all_cf_logits.topk(k=all_cf_logits.shape[-1], dim=-1)
                        # else:
                        #     top10_v, top10_idx = all_cf_logits.topk(k=10, dim=-1)
                        
                        token_sums = scatter_sum((all_cf_logits / beta).softmax(dim=-1) , all_tokens, dim=-1, dim_size=logits.size(-1))
                        # token_sums = torch.zeros(logits.size(-1), device=device)
                        # # 获取每个item_id的CF logit值
                        # item_cf_values = cf_logits[batch_idx, all_item_ids]
                        # # 使用scatter_sum将相同token的值累加
                        # token_sums.scatter_add_(0, all_tokens, item_cf_values)
                        
                        # 将结果存储到token_cf_logits中
                        token_cf_logits[batch_idx, generated_start_idx + t - 1] = token_sums
            else:
                # If path is invalid, only allow EOS token
                continue
        
        if self.args.use_msl:
            # 将不允许的token的logits设置为负无穷
            logits = logits.masked_fill(~allowed_tokens_mask, float('-inf'))
            # 将token_cf_logits中不允许的token的值设置为负无穷
            token_cf_logits = token_cf_logits.masked_fill(~allowed_tokens_mask, 0)
        shift_token_cf_logits = token_cf_logits[..., :-1, :].contiguous()
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels != -100

        shift_labels = shift_labels[mask]
        shift_logits = shift_logits[mask]
        
        # Check if shift_token_cf_logits exists and use it as a soft label
        # Extract the masked shift_token_cf_logits
        masked_shift_token_cf_logits = shift_token_cf_logits[mask]
        
        # Normalize the cf_logits using softmax
        # normalized_cf_logits = F.softmax(masked_shift_token_cf_logits / tau_cf, dim=-1)
        normalized_cf_logits = masked_shift_token_cf_logits
        # Create one-hot encoding for shift_labels
        one_hot_labels = F.one_hot(shift_labels, num_classes=shift_logits.size(-1)).to(shift_logits.dtype)
        
        # Combine one-hot labels with normalized CF logits as soft labels
        alpha = self.alpha  # Weight parameter for balancing between labels and CF logits
        soft_labels = (1 - alpha) * one_hot_labels + alpha * normalized_cf_logits
        
        tau = self.tau
        all_logits = torch.exp(shift_logits / tau)
        pos_loss = -torch.log((all_logits * soft_labels).sum(dim=-1))
        neg_loss = torch.log(all_logits.sum(dim=-1))
        loss = (pos_loss + neg_loss) 

        loss = loss.mean()
        decoder_outputs.loss = loss
        return decoder_outputs

    def wrap_embs(self, input_ids, user_idx,  hist_idx):
        inputs_embeds = self.model.get_input_embeddings()(input_ids).requires_grad_(False)
        batch_size = input_ids.shape[0]
        if self.args.use_feature:
            # if title_idx is None:
            #     all_item_ids = prefer_idx
            # else:    
            all_item_ids = torch.cat([hist_idx], dim=-1)
            user_embs = self.mlp(self.cf_model.get_user_embs(user_idx, hist_idx)).to(inputs_embeds.dtype)
            all_item_embs = self.mlp(self.cf_model.get_item_embs(all_item_ids))
            all_item_embs = all_item_embs.to(dtype=inputs_embeds.dtype)

            for i in range(batch_size):
                if (input_ids[i] == self.user_token_id).sum() == 1:
                    inputs_embeds[i][input_ids[i] == self.user_token_id] = user_embs[i]
                row_seq_len = (all_item_ids[i] != self.padding_item_id).sum()
                
                inputs_embeds[i][input_ids[i] == self.item_token_id] = all_item_embs[i][-row_seq_len:]
        return inputs_embeds

    def generate(self, input_ids, user_id, hist_id, attention_mask, temperature=0.8, do_sample=False, num_beams=5, max_gen_length=64, min_gen_length=1, repetition_penalty=1.0, length_penalty=1.0, num_return_sequences=20):
        
        if num_beams == 1 and not do_sample and num_return_sequences > 1:
            num_beams = num_return_sequences
            print(f"Warning: Greedy search does not support num_return_sequences > 1. Switching to beam search with num_beams={num_beams}")
        
        # Create a custom logits processor that only applies trie constraints
        class TrieLogitsProcessor:
            def __init__(self, parent):
                self.parent = parent

            def __call__(self, input_ids, scores):
                
                
                # Apply trie constraints
                batch_size = input_ids.shape[0]
                vocab_size = scores.shape[-1]
                allowed_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
                
                for batch_idx in range(batch_size):
                    # Get the start of generated tokens (after the last BOS token)
                    generated_start_idx = (input_ids[batch_idx] == self.parent.tokenizer.bos_token_id).nonzero(as_tuple=True)[0][-1] + 1
                    generated_tokens = input_ids[batch_idx, generated_start_idx:]
                    # 使用Trie类的缓存机制获取节点
                    node, valid_path = self.parent.trie.get_node_for_sequence(batch_idx, generated_tokens)
                    
                    if valid_path:
                        if node.is_end_of_word or len(node.children) == 0:
                            # Only allow EOS token
                            allowed_tokens_mask[batch_idx, self.parent.tokenizer.eos_token_id] = True
                        else:
                            # Allow all valid next tokens from trie
                            allowed_tokens_mask[batch_idx, list(node.children.keys())] = True
                    else:
                        # If path is invalid, only allow EOS token
                        allowed_tokens_mask[batch_idx, self.parent.tokenizer.eos_token_id] = True
                
                scores = scores.masked_fill(~allowed_tokens_mask, float('-inf'))
                return scores
        num_return_sequences = min(num_return_sequences, num_beams) if not do_sample else num_return_sequences
        # Generate sequences
        inputs_embs = self.wrap_embs(input_ids, user_id, hist_id)
        outputs = self.model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embs,
            attention_mask=attention_mask,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True,
            logits_processor=[TrieLogitsProcessor(self)],
        )
        
        # Get only the newly generated tokens (excluding input sequence)
        input_length = input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        
        # Get sequence scores directly from model outputs
        if hasattr(outputs, 'sequences_scores'):
            sequence_scores = outputs.sequences_scores
        else:
            # If sequences_scores not available, calculate raw scores from token scores
            sequence_scores = torch.zeros(len(generated_tokens), device=generated_tokens.device)
            for i, score in enumerate(outputs.scores):
                selected_tokens_scores = score[torch.arange(len(generated_tokens)), generated_tokens[:, i]]
                sequence_scores += selected_tokens_scores
        
        # Process each sequence and get scores
        batch_size = generated_tokens.shape[0] // num_return_sequences
        all_results = []
        
        for batch_idx in range(batch_size):
            batch_start = batch_idx * (num_return_sequences)
            batch_end = (batch_idx + 1) * (num_return_sequences)
            
            batch_sequences = generated_tokens[batch_start:batch_end]
            batch_scores = sequence_scores[batch_start:batch_end]
            
            batch_results = []
            for seq, score in zip(batch_sequences, batch_scores):
                # Find where the sequence ends (either EOS token or end of sequence)
                eos_positions = (seq == self.tokenizer.eos_token_id).nonzero()
                if len(eos_positions) > 0:
                    # If EOS token is found, only take tokens up to EOS
                    seq_length = eos_positions[0].item()
                    seq = seq[:seq_length]
                else:
                    seq_length = len(seq)
                
                # Decode the sequence
                text = self.tokenizer.decode(seq, skip_special_tokens=True)
                if len(self.tokenizer.pad_token) == 1:
                    text = text.strip(self.tokenizer.pad_token)
                # Try to map the generated text to an item ID
                try:
                    batch_results.append((text, score.item()))
                except KeyError:
                    continue
            
            # Sort by scores and take top 20 unique items
            
            all_results.append(batch_results)
        
        # Unzip the results
        processed_outputs = [[result[0] for result in batch] for batch in all_results]
        
        return processed_outputs

    def merge(self,outputs):
        # if dist.is_initialized():
        #     all_rank_outputs = [None for _ in range(dist.get_world_size())]    
        #     dist.all_gather_object(all_rank_outputs,outputs)
        #     outputs = [x for y in all_rank_outputs for x in y] ## all_rank_output[i]: i-th batch output
        single_batch_output_cnt = len(outputs[0])
        ret = [[] for _ in range(single_batch_output_cnt)]
        for idx in range(single_batch_output_cnt):
            for batch in outputs:
                ret[idx].append(batch[idx])
        return ret

    def training_step(self, batch, batch_idx):
        # 打印训练数据查看数据分片是否生效
        # if batch_idx == 0:
        #     logger.info(f"train pid: {os.getpid()} batch: {batch}")

        # 定义train逻辑
        output = self(
            batch_idx=batch_idx,
            user_idx=batch['user_idx'],
            title_idx=batch['title_idx'],
            input_ids=batch['input_ids'],
            hist_id=batch['hist_id'],
            attention_mask=batch['attention_mask'],
            is_testing = False,
        )
        loss = output.loss
        
        self.log("loss", loss.item(), prog_bar=True, sync_dist=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Generate item titles and ids
        generated_titles = self.generate(
            input_ids=batch['input_ids'],
            user_id=batch['user_idx'],
            hist_id=batch['hist_id'],
            attention_mask=batch['attention_mask'],
            num_beams = max(self.valid_Ks)
        )
        
        # Store the ground truth and generated results for metric calculation
        for i, (target_id, gen_title) in enumerate(zip(batch['title_idx'], generated_titles)):
            self.val_info.append({
                'target_id': target_id.item(),
                'generated_title': gen_title,
                'user_id': batch['user_idx'][i].item()
            })
        
        return {}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        
        idx = batch['data_idx']
        generated_titles = self.generate(
            input_ids=batch['input_ids'],
            user_id=batch['user_idx'],
            hist_id=batch['hist_id'],
            attention_mask=batch['attention_mask'],
            num_beams=max(self.test_Ks),
        )
        
        # Store the ground truth and generated results for metric calculation
        for i, (target_id, gen_title) in enumerate(zip(batch['title_idx'],generated_titles)):
            self.val_info.append({
                'target_id': target_id.item(),
                'target_title': self.item_dict[target_id.item()],
                'generated_title': gen_title,
                'user_id': batch['user_idx'][i].item(),
                'data_idx': idx[i].item()
            })
        return {}


    def calculate_metrics(self, results, Ks):
        """Calculate Recall, NDCG and Hit Ratio at K"""
        metrics = {
            'recall': np.zeros(len(Ks)),
            'ndcg': np.zeros(len(Ks)),
            'hit_ratio': np.zeros(len(Ks))
        }
        
        total_samples = len(results)
        
        for result in results:
            target_id = result['target_id']
            target_title = self.item_dict[target_id]
            pred_titles = result['generated_title']
            
            # Convert predictions to binary relevance list
            r = [0] * max(Ks)
            for m, pred_title in enumerate(pred_titles):
                if pred_title == target_title:
                    r[m] = 1
            
            for i, K in enumerate(Ks):
                # Calculate Recall@K
                if sum(r[:K]) > 0:
                    metrics['recall'][i] += 1
                
                # Calculate NDCG@K
                dcg = 0
                idcg = 1  # ideal DCG for single relevant item
                for j in range(K):
                    if r[j] == 1:
                        dcg += 1 / np.log2(j + 2)
                metrics['ndcg'][i] += dcg / idcg
                
                # Calculate Hit@K
                metrics['hit_ratio'][i] += np.sum(r[:K]) > 0
        
        # Normalize metrics
        for metric in metrics:
            metrics[metric] = (metrics[metric] / total_samples).tolist()
        
        return metrics

    def on_validation_epoch_end(self):
        Ks = self.valid_Ks
        
        # Calculate metrics
        metrics = self.calculate_metrics(self.val_info, Ks)
        
        # Log metrics
        for i, K in enumerate(Ks):
            self.log(f"val_hit@{K}", metrics['hit_ratio'][i], prog_bar=True, sync_dist=True)
        
        # Clear validation info
        self.val_info.clear()
        
        return metrics

    def on_test_epoch_end(self):
        # Calculate and save metrics
        Ks = self.test_Ks
        metrics = self.calculate_metrics(self.val_info, Ks)
        
        # Create results dictionary
        results = {
            'metrics': metrics,
            'raw_predictions': {row['data_idx']: row for row in self.val_info}
        }
        
        # Save results to JSON file
        output_path = os.path.join(self.args.dirpath, f"{self.args.dataset}_{self.args.llm_model}_test_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Log metrics
        for i, K in enumerate(Ks):
            self.log(f"test_ndcg@{K}", metrics['ndcg'][i], prog_bar=True, sync_dist=True)
            self.log(f"test_hit@{K}", metrics['hit_ratio'][i], prog_bar=True, sync_dist=True)
        
        # Clear test info
        self.val_info.clear()
        return metrics

    def on_save_checkpoint(self, checkpoint):
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)
    
    def setup(self, stage=None) -> None:
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        self.n_user = self.trainer.datamodule.n_user
        self.n_item = self.trainer.datamodule.n_item
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()
        # self.interact_data = torch.LongTensor(self.trainer.datamodule.interact_data[['uid', 'iid']].to_numpy())
        
        self.sample_num = len(train_loader.dataset)
        print("self.sample_num :", self.sample_num)

        # Calculate total steps
        tb_size = train_loader.batch_size * max(1, self.trainer.num_devices)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) * ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                # 根据p.requires_grad判断参数是否需要更新
                "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                # 根据p.requires_grad判断参数是否需要更新
                "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        num_warmup_steps = int(self.total_steps*self.hparams.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=20,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]