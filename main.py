# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.

import argparse
import os
import traceback
from transformers import  AutoTokenizer
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from data import create_data_module 
import json
import torch

def main(args):
    seed_everything(args.seed)

    ### 只有msl才会设置tau
    if not args.use_msl:
        args.tau = 1
    if args.ckpt_path is not None:
        dirpath = os.path.join(*args.ckpt_path.split("/")[:-1])
        config_file = os.path.join(dirpath, "config.json")
        args = overwrite(args, config_file)
        args.dirpath = dirpath
    print(args)

    model_dict = {
        "llama3": "xxxx",
        "llama3-3b": "xxxx",
        "llama2": "xxxx",
        "qwen0.5B": "xxxx",
        "qwen1.5B": "xxxx",
    }


    if args.llm_model in model_dict.keys():
        args.model_name_or_path = model_dict[args.llm_model]
    else:
        raise Exception("not support llm models")
    model_path = f"checkpoints/llm_model={args.llm_model}-cf_model={args.cf_model}-dataset={args.dataset}-model={args.model}-alpha={args.alpha}-tau={args.tau}-seed={args.seed}-use_msl={args.use_msl}-use_feature={args.use_feature}"
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    if args.ckpt_path is None:
        args.dirpath = model_path
        with open(os.path.join(model_path, "config.json"), "w") as f:
            f.write(json.dumps(vars(args), indent=4))

    gradient_accumulation_steps = args.train_batch_size // args.micro_batch_size
    # data
    tokenizer = AutoTokenizer.from_pretrained (args.model_name_or_path)

    if "qwen" in args.llm_model:
        tokenizer.bos_token_id = tokenizer("<|im_start|>")['input_ids'][0]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
    tokenizer.padding_side = args.padding_side
    if args.use_feature:
        special_tokens = ["[UserEmb]" , "[ItemEmb]"]
        tokenizer.add_tokens(special_tokens)
    
    datamodule = create_data_module(args, tokenizer)

    args.idx_num = datamodule.n_item
    # model
    
    if args.model =="llm4rec":
        from models.LLM4Rec import LLM4Rec
        model = LLM4Rec(args=args,tokenizer=tokenizer, model_name_or_path=args.model_name_or_path,n_user=datamodule.n_user, n_item=datamodule.n_item, datamodule=datamodule)
    else:
        raise Exception("no suitable models")

    
    if args.ckpt_path is None:
        # 添加多机多卡环境适配及模型保存支持oss与pangu
        # plugins = [AntClusterEnvironment(), AntCheckpointIO()]

        # 模型保存的callback
        from pytorch_lightning.callbacks import ModelCheckpoint
        save_model = True
        eval_every_epoch = 2
        epoch = args.epoch 
        metric = "val_hit@3"

        if args.sample_num == -1 or  args.sample_num >= 10000:
            epoch = 10
            eval_every_epoch = 2
        # 自定义checkpoint策略
        ckpt_callback = ModelCheckpoint(
            dirpath=model_path,
            # save_last=True,
            monitor=metric,
            every_n_epochs=eval_every_epoch,
            save_top_k=1,
            mode="max"
        )
        
        from pytorch_lightning.callbacks import EarlyStopping
        early_stop_callback = EarlyStopping(
            monitor=metric,
            min_delta=0.001,
            patience=3,
            verbose=False,
            mode='max',
            )

        callbacks = [ckpt_callback, early_stop_callback]

        trainer = Trainer(
            max_epochs=epoch,
            # devices=args.num_gpus,
            devices=args.num_gpus,
            accelerator='gpu',
            # strategy='ddp_find_unused_parameters_true',
            num_nodes=args.num_nodes,
            enable_checkpointing=save_model,
            accumulate_grad_batches=gradient_accumulation_steps,
            # plugins=plugins,
            callbacks=callbacks,
            # fast_dev_run = True,
            enable_progress_bar=True,
            logger=False,
            # limit_val_batches=10,  # no eval
            # val_check_interval=1,
            check_val_every_n_epoch=eval_every_epoch
        )

        # logger.info("start train #########################")
        if args.mode in ["inference", "generation"]:
            trainer.test(model, datamodule)
        else:
            trainer.fit(model, datamodule)
            ckpt_path = trainer.checkpoint_callback.best_model_path
            args.ckpt_path = ckpt_path
            ckpt = torch.load(args.ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['state_dict'], strict=False)
            trainer.test(model, datamodule)
    else:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        trainer = Trainer(
            max_epochs=args.epoch,
            devices=args.num_gpus,
            accelerator='gpu',
            strategy='ddp',
            num_nodes=args.num_nodes,
            enable_checkpointing=True,
            # plugins=plugins,
            # fast_dev_run=True,
            enable_progress_bar=True,
            logger=False,
        )

        # logger.info("start train #########################")
        # trainer.test(model, ckpt_path=args.ckpt_path, datamodule=datamodule)
        trainer.test(model,  datamodule=datamodule)

def overwrite(args, config_file):
    if os.path.exists(config_file):
        with open(config_file) as f:
            config = json.load(f)
            args.model = config['model']
            if not args.dataset:
                args.dataset = config['dataset']
            args.lora_r = config['lora_r']
            args.lora_alpha = config['lora_alpha']
            args.lora_target_modules = config['lora_target_modules']
            args.lora_dropout = config['lora_dropout']
            args.max_item_len = config['max_item_len']
            args.max_seq_length = config['max_seq_length']
            args.model_name_or_path = config['model_name_or_path']
            if "use_feature" in config:
                args.use_feature = config['use_feature']
            if "llm_model" in config:
                args.llm_model = config['llm_model']
            if "alpha" in config:
                args.alpha = config['alpha']
            if "cf_model" in config:
                args.base_model = config['cf_model']
            if "sample_num" in config:
                args.sample_num = config['sample_num']
            if "save" in config:
                args.save = config['save']
            else:
                args.save = "all"
            if 'padding_side' in config:
                args.padding_side = config['padding_side']
    return args



parser = argparse.ArgumentParser(description='llm_seq')
# GPU配置
parser.add_argument("--ckpt_path", type=str, default=None)
parser.add_argument('--num_nodes', default=1, type=int)
parser.add_argument('--num_gpus', default=1, type=int)
parser.add_argument("--mode", default="train", choices=["train", "inference"])
parser.add_argument('--data_path', default='data/', type=str)
parser.add_argument('--dataset', default="", type=str)
parser.add_argument("--max_item_len", default=10, type=int)
parser.add_argument("--model", default="llm4rec")
parser.add_argument("--cf_model", default="sasrec", type=str, help=["sasrec, lightgcn"])
parser.add_argument("--use_feature", default=False, action="store_true")
# 训练配置
parser.add_argument("--llm_model", default="llama3-3b", type=str)
parser.add_argument('--model_name_or_path', default="/workspace/xiaotu/models/meta-llama__Meta-Llama-3.1-8B-Instruct/", type=str) #bigscience/bloomz-7b1-mt meta-llama/Llama-2-7b-chat-hf THUDM/chatglm2-6b
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--micro_batch_size', default=4, type=int)
parser.add_argument('--val_batch_size', default=4, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--warmup_ratio', default=0.01, type=float)
parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--max_seq_length', type=int, default=2048)
# 模型ckpt保存目录
parser.add_argument('--model_path', default=f'/workspace/xiaotu/checkpoints', type=str)
# 数据配置
parser.add_argument('--lora_r', type=int, default=8)
parser.add_argument('--lora_dropout', type=float, default=0.05)
parser.add_argument('--lora_alpha', type=int, default=16)
parser.add_argument('--lora_target_modules', type=list, default=['v_proj', 'q_proj', 'k_proj', "o_proj"])
parser.add_argument("--alpha",type=float, default=0.0)
parser.add_argument("--attention", default="last", help="[all, prefer, last]")
parser.add_argument("--save", default="part", help="[part, all]")
parser.add_argument("--padding_side", default="left", type=str)
parser.add_argument("--sample_num", default=10000, type=int)
parser.add_argument("--tau", default=4.5, type=float)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--use_msl", default=False, action="store_true")
args = parser.parse_args()
main(args)