import numpy as np
import torch


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
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, device, maxlen, hidden_units):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = device
        self.maxlen = maxlen
        num_heads = 1
        num_layers = 2
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(maxlen, hidden_units, padding_idx=0)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
        for _ in range(num_layers):
            new_attn_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(hidden_units,
                                                            num_heads, 0)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, 0)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        device = log_seqs.device
        seqs = self.item_emb(log_seqs)  # batch_size x max_len x embedding_dim
        seqs *= self.item_emb.embedding_dim**0.5

        single_seq = torch.arange(log_seqs.shape[1], dtype=torch.long, device=device)
        positions = single_seq.unsqueeze(0).repeat(log_seqs.shape[0], 1)
        seqs += self.pos_emb(positions)

        # seqs = self.emb_dropout(seqs)  # 使得embedding中某些元素随机归0

        timeline_mask = log_seqs == 0  # batch_size x max_len
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim ；True表示序列中不为padding

        tl = seqs.shape[1]  # time dim len for enforce causality
        # 即判断第i个item是否对第j个item起作用，仅当j>=i时起作用; 返回一个上方为1的上三角矩阵
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=device))

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

        return log_feats, attn_output_weights[:, -1, :]

    def get_attention(self, log_seqs):
        """
        Note the SASRec sequence is left padding, 
        i.e. filling the pad from the left side, 
        e.g. [0, 0, 0, 43, 61, 4]
        """
        log_feats, attentions = self.log2feats(log_seqs) # user_ids hasn't been used yet
        ## (batch_size, layers, item_size)
        return attentions
    def get_user_embs(self, user_ids, log_seqs):

        log_feats, attentions = self.log2feats(log_seqs)
        return log_feats[:, -1, :]
    def get_item_embs(self, log_seqs):
        return self.item_emb(log_seqs)