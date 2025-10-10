from collections import Counter
import copy
import pdb
import numpy as np
import pandas as pd
import ast
from torch.utils.data import Dataset
import world
import torch.nn.functional as F
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import random
import json

def read_file(path, user_history_dict=None, maxlen=10):
    df = pd.read_csv(path)
    df["user_id"] = df["uid"].astype(int)
    
    # Initialize user_history_dict if it's None
    if user_history_dict is None:
        user_history_dict = {}
    
    # Process each row to build and update user history
    item_ids_list = []
    for row in df.to_dict("records"):
        user_id = row["user_id"]
        current_item = row["iid"]
        
        # For first-time users, initialize history with hist_id
        if user_id not in user_history_dict:
            user_history_dict[user_id] = ast.literal_eval(row["hist_id"])
        
        # Get current history for this user
        current_history = ast.literal_eval(row["hist_id"]) # user_history_dict[user_id].copy()[-maxlen:]
        item_ids_list.append(current_history)
        
        # Update history with current item for future interactions
        user_history_dict[user_id].append(current_item)
    
    # Add the processed item_ids to the dataframe
    df["item_ids"] = item_ids_list
    
    return df, user_history_dict


class Data_Pro:
    def __init__(self, dataroot):
        train_file = os.path.join(dataroot, "train.csv")
        valid_file = os.path.join(dataroot, "valid_5000.csv")
        test_file = os.path.join(dataroot, "test_5000.csv")

        with open(os.path.join(dataroot, "data_stat.json")) as f:
            data_stat = json.load(f)

        # Initialize user history dictionary
        user_history_dict = {}
        
        # Process train, valid, and test files while maintaining user history
        train_df, user_history_dict = read_file(train_file, user_history_dict, world.config["maxlen"])
        valid_df, user_history_dict = read_file(valid_file, user_history_dict, world.config["maxlen"])
        test_df, user_history_dict = read_file(test_file, user_history_dict, world.config["maxlen"])
        
        # Store the dataframes
        self.train_df, self.valid_df, self.test_df = train_df, valid_df, test_df
        
        # Store the user history dictionary for potential future use
        self.user_history_dict = user_history_dict

        max_item_id = data_stat['num_item']

        self.item_num = max_item_id
        print("Max Item ID: ", self.item_num)
        print("Max User ID: ", data_stat['num_user'])

    def get_data_df(self):
        return self.train_df, self.valid_df, self.test_df


class SeqDataset(Dataset):
    def __init__(self, data_frame, padding_side, max_len=world.config["maxlen"]):
        self.df = data_frame
        self.padding_side = padding_side
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item_ids = row["item_ids"].copy()
        if len(item_ids) > self.max_len:
            item_ids = item_ids[-self.max_len :]
        item_ids = torch.tensor(item_ids, dtype=torch.long) if type(item_ids) is list else item_ids
        seq, label = item_ids, row['iid']

        if self.padding_side == "right":
            seq = F.pad(seq, (0, self.max_len - seq.size(0)), "constant", 0)
        elif self.padding_side == "left":
            seq = F.pad(seq, (self.max_len - seq.size(0), 0), "constant", 0)

        return seq, label

class SeqCTRDataset(Dataset):
    def __init__(self, data_frame, padding_side, max_len=world.config["maxlen"]):
        self.df = data_frame
        self.padding_side = padding_side
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item_ids = row["item_ids"].copy()
        if len(item_ids) > self.max_len:
            item_ids = item_ids[-self.max_len :]
        item_ids = torch.tensor(item_ids, dtype=torch.long) if type(item_ids) is list else item_ids
        seq, pos, label = item_ids, row['iid'], row['label']

        if self.padding_side == "right":
            seq = F.pad(seq, (0, self.max_len - seq.size(0)), "constant", 0)
        elif self.padding_side == "left":
            seq = F.pad(seq, (self.max_len - seq.size(0), 0), "constant", 0)

        return seq, pos, label
