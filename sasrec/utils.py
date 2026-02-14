import math
import pdb
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter, defaultdict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def evaluate(model, dataloader: DataLoader, sample_total: int, mask_id=None, topk_list=[1, 5, 10, 20]):
    NDCG, HR = [], []
    NDCG_total, HR_total = torch.zeros(size=(len(topk_list),)), torch.zeros(size=(len(topk_list),))

    with torch.no_grad():
        for seq, pos in dataloader:
            seq, pos = seq.cuda(), pos.cuda()
            target_item_ids = pos

            logits = -model(seq)

            rank_tensor = logits.argsort(dim=-1).argsort(dim=-1)
            target_item_ranks = rank_tensor[torch.arange(rank_tensor.size(0)), target_item_ids]
            rank_list_tensor = target_item_ranks

            for i, k in enumerate(topk_list):
                Hit_num = (rank_list_tensor < k).sum().item()
                HR_total[i] += Hit_num

                mask = rank_list_tensor < k
                NDCG_num = 1 / torch.log(rank_list_tensor[mask] + 2)
                NDCG_num = NDCG_num.sum().item()
                NDCG_total[i] += NDCG_num

    NDCG = NDCG_total / (sample_total * (1.0 / math.log(2)))
    HR = HR_total / sample_total

    result_dict = dict()
    for i in range(len(topk_list)):
        result_dict["NDCG@" + str(topk_list[i])] = round(NDCG[i].item(), 4)
    for i in range(len(topk_list)):
        result_dict["HR@" + str(topk_list[i])] = round(HR[i].item(), 4)
    result_dict["default"] = result_dict["NDCG@10"]

    return result_dict


def get_negative_items(seq, pos, item_num):
    pos = pos.unsqueeze(-1)
    seq_pos = torch.cat([seq, pos], dim=1)

    probabilities = torch.ones(size=(seq.shape[0], item_num + 1), dtype=torch.float, device="cuda")
    batch_indices = torch.arange(seq.shape[0], device="cuda").view(-1, 1)
    probabilities[batch_indices, seq_pos] = 0
    probabilities[:, 0] = 0
    neg = torch.multinomial(probabilities, 1, replacement=False)

    return seq.long(), pos.long(), neg.long()


def calcu_propensity_score(pos, item_num):
    freq = Counter(list(pos))
    for i in range(item_num + 1):
        if i not in freq.keys():
            freq[i] = 0
    pop = [freq[i] for i in range(item_num + 1)]
    pop = np.array(pop)
    ps = pop + 1
    ps = ps / np.sum(ps)
    ps = np.power(ps, 0.05)
    return ps
