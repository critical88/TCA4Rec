import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import world
import utils
from torch.nn import TransformerEncoderLayer
import os

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, item_num):
        super(SASRec, self).__init__()

        self.item_num = item_num
        self.config = world.config

        self.item_emb = torch.nn.Embedding(self.item_num + 1, self.config["hidden_units"], padding_idx=0)
        self.pos_emb = torch.nn.Embedding(self.config["maxlen"], self.config["hidden_units"])

        self.emb_dropout = torch.nn.Dropout(p=self.config["dropout_rate"])
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.config["hidden_units"], eps=1e-8)

        for _ in range(self.config["num_blocks"]):
            new_attn_layernorm = torch.nn.LayerNorm(self.config["hidden_units"], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(
                self.config["hidden_units"], self.config["num_heads"], self.config["dropout_rate"]
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.config["hidden_units"], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.config["hidden_units"], self.config["dropout_rate"])
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def log2feats(self, log_seqs, return_attention=False):
        seqs = self.item_emb(log_seqs)  # batch_size x max_len x embedding_dim
        seqs *= self.item_emb.embedding_dim**0.5

        single_seq = torch.arange(log_seqs.shape[1], dtype=torch.long, device="cuda")
        positions = single_seq.unsqueeze(0).repeat(log_seqs.shape[0], 1)
        seqs += self.pos_emb(positions)

        seqs = self.emb_dropout(seqs)  # 使得embedding中某些元素随机归0

        timeline_mask = log_seqs == 0  # batch_size x max_len
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim ；True表示序列中不为padding

        tl = seqs.shape[1]  # time dim len for enforce causality
        # 即判断第i个item是否对第j个item起作用，仅当j>=i时起作用; 返回一个上方为1的上三角矩阵
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device="cuda"))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, attn_output_weights = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )  # query key value

            # mha_outputs.shape 10 x 256 x 64
            # attn_output_weights.shape 256 x 10 x 10 最后一行代表序列前面的item对最后一个item的权重

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        if return_attention:
            return log_feats, attn_output_weights[:, -1, :]
        else:
            return log_feats

    def forward(self, log_seqs):  # for training
        log_feats = self.log2feats(log_seqs)  # batch_size x max_len x embedding_dim
        final_feat = log_feats[:, -1, :].squeeze(dim=1)  # batch_size x embedding_dim

        # log_feats = F.normalize(log_feats)
        item_embs = self.item_emb.weight  # item_num x embedding_dim
        # item_embs = F.normalize(item_embs)
        logits = torch.matmul(final_feat, item_embs.t())  # batch_size x item_num

        return logits

    def compute_loss(self, batch):
        """残血版SASRec"""
        seq, pos = batch
        seq, pos = seq.cuda(), pos.cuda()

        seq, pos, neg = utils.get_negative_items(seq, pos, self.item_num)
        logits = self.forward(seq)

        pos_logits = torch.gather(logits, 1, pos)
        neg_logits = torch.gather(logits, 1, neg)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device="cuda"), torch.zeros(
            neg_logits.shape, device="cuda"
        )

        # indices = pos != 0
        logits = torch.cat((pos_logits, neg_logits), 0)
        labels = torch.cat((pos_labels, neg_labels), 0)

        bce_criterion = torch.nn.BCEWithLogitsLoss()
        loss = bce_criterion(logits, labels)

        return loss

    def compute_loss_full(self, batch):
        """满血版SASRec"""
        seq, pos = batch
        seq, pos = seq.cuda(), pos.cuda()

        log_feats = self.log2feats(seq)

        cat_seq_pos = torch.cat([seq, pos.unsqueeze(1)], dim=1)
        seq, pos, neg = utils.get_negative_items_full(cat_seq_pos, self.item_num)

        pos_embs = self.item_emb(pos)
        neg_embs = self.item_emb(neg)  # bacth_size x maxlen x neg_num x emb_dim
        # pos_embs = F.normalize(pos_embs)
        # neg_embs = F.normalize(neg_embs)
        # log_feats = F.normalize(log_feats)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        pos_labels = torch.ones(pos_logits.shape, device="cuda")
        neg_labels = torch.zeros(neg_logits.shape, device="cuda")

        indices = pos != 0
        bce_criterion = torch.nn.BCEWithLogitsLoss()
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])

        return loss

    def save_item_embeddings(self, file_path):
        np.save(file_path, self.item_emb.weight.data.detach().cpu().numpy() )
