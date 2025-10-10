import os
import multiprocessing

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="SASRec")
    parser.add_argument("--model", type=str, default="SASRec")
    parser.add_argument("--dataset", type=str)

    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=201)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)

    parser.add_argument("--maxlen", type=int, default=10, help="input (X) lens")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--state_dict_path", default=None, type=str)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--label", action="store_true", default=False)
    
    # SASRec超参数
    parser.add_argument("--enable_full_SASRec", type=int, default=0)
    
    # BERT超参数
    parser.add_argument("--mask_prob", type=float, default=0.5, help="mask_prob")
    
    # DROS超参数
    parser.add_argument("--enable_DROS", type=int, default=0)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0)
    return parser.parse_args()


args = parse_args()
config = vars(args)

CORES = multiprocessing.cpu_count() // 2

seed = args.seed


def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")
