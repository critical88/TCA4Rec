import json
import os
import random

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"

import time
import torch

from tqdm import tqdm
import copy
# import nni

from model import SASRec
import world
from logger import CompleteLogger
import utils
from os.path import join
from pprint import pprint

from dataloader import SeqDataset, Data_Pro
from torch.utils.data import DataLoader

if not "NNI_PLATFORM" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = world.config["cuda"]
else:
    import nni
    optimized_params = nni.get_next_parameter()
    world.config.update(optimized_params)

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

dataset = world.config["dataset"]
dataroot = os.path.join("../data/", dataset)
model_name = world.config["model"]
logroot = os.path.join("./log", model_name, dataset)

# log保存路径
if "NNI_PLATFORM" in os.environ:
    save_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], "tensorboard")
else:
    save_dir = logroot
    save_dir = join(logroot, time.strftime("%m-%d-%Hh%Mm%Ss"))
    i = 0
    while os.path.exists(save_dir):
        new_save_dir = save_dir + str(i)
        i += 1
        save_dir = new_save_dir
    logger = CompleteLogger(root=save_dir)

pprint(world.config)


if __name__ == "__main__":
    # 数据处理
    data_pro = Data_Pro(dataroot)
    train_df, valid_df, test_df = data_pro.get_data_df()
    item_num = data_pro.item_num

    model_name = world.config["model"]
    padding_side = "left" 
    batch_size = world.config["batchsize"]

    if model_name in ["SASRec"]:
        
        train_dataset = SeqDataset(train_df, padding_side, world.config["maxlen"])
        valid_dataset = SeqDataset(valid_df, padding_side, world.config["maxlen"])
        test_dataset = SeqDataset(test_df, padding_side, world.config["maxlen"])

    train_dataloader, valid_dataloader, test_dataloader = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16),
        DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=16),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16),
    )

    # 模型
    
    if model_name == "SASRec":
        model = SASRec(item_num).cuda()

    state_dict_path = world.config["state_dict_path"]
    if state_dict_path is not None:
        model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))

    adam_optimizer = torch.optim.Adam(
        model.parameters(), lr=world.config["lr"], weight_decay=world.config["weight_decay"]
    )


    start_total = time.time()
    patience = 0
    best_NDCG = 0
    step = 0
    best_metric = 0
    metric = "NDCG@10"

    best_model_dict = None
    for epoch in range(world.config["num_epochs"]):
        print("===================================")
        print("Start Training Epoch {}".format(epoch))

        # Train
        model.train()
        for batch in tqdm(train_dataloader):
            loss = model.compute_loss(batch)

            
            adam_optimizer.zero_grad()
            loss.backward()
            adam_optimizer.step()

            # w.add_scalar(f"Train/Loss", loss, step)
            step += 1

        # Test
        if epoch % 1 == 0:
            model.eval()
            t_valid = utils.evaluate(model, valid_dataloader, len(valid_dataset))
            
            # write_tensorboard_metric(w, t_valid, epoch, "Valid")
            print("Valid\n", t_valid)
            if "NNI_PLATFORM" in os.environ:
                nni.report_intermediate_result(t_valid)

            if t_valid[metric] > best_metric:
                best_metric = t_valid[metric]
                t_test = utils.evaluate(model, test_dataloader, len(test_dataset))
                # write_tensorboard_metric(w, t_test, epoch, "Test")
                print("Test\n", t_test)
                patience = 0
                best_model_dict = copy.deepcopy(model.state_dict())
            else:
                patience += 1
                print("Patience{}/10".format(patience))
                if patience >= 10:
                    break

    if "NNI_PLATFORM" not in os.environ:
        torch.save(best_model_dict, os.path.join(save_dir, "best_model.pth"))
    model.load_state_dict(best_model_dict)

    model.eval()
    t_test = utils.evaluate(model, test_dataloader, len(test_dataset))

    print("The Finial Test Metric for Model is following: ")
    print(t_test)
    if "NNI_PLATFORM" in os.environ:
        nni.report_final_result(t_test)

    # w.close()
    print("Total time:{}".format(time.time() - start_total))
    
    # Save item embeddings for all model types
    # model_name = world.config["model"]
        
    # # Create directory if it doesn't exist
    # embedding_dir = f'log/{model_name}/{world.config["dataset"]}'
    # if not os.path.exists(embedding_dir):
    #     os.makedirs(embedding_dir)
        
    # # Save item embeddings
    # embedding_path = os.path.join(embedding_dir, "item_embeddings.npy")
    # model.save_item_embeddings(embedding_path)
    # print(f"Item embeddings saved to {embedding_path}")

    print("Training Done!")
