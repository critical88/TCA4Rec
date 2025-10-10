import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np
import argparse
from collections import defaultdict
import jsonlines
import torch
RANDOM_SEED = 42

def partition(rating_df):
    """
    8:1:1
    """
    rating_len = len(rating_df)

    train_size = int(rating_len * 0.8)
    val_size = rating_len // 10
    train_data = rating_df[:train_size]
    val_data = rating_df[train_size: (train_size + val_size)]
    test_data = rating_df[(train_size + val_size):]

    return train_data, val_data, test_data
def print_stat(rating_df, filtered=False):
    print(("filtered" if filtered else "#") + " user:" + str(len(rating_df['uid'].unique())))
    print(("filtered" if filtered else "#") + " item:" + str(len(rating_df['iid'].unique())))
    print(("filtered" if filtered else "#") + " interaction is:" + str(len(rating_df)))


def save_item_data(item_df, target_data_path):
    item_file = os.path.join(target_data_path, "item.csv")

    if not os.path.exists(target_data_path):
        os.makedirs(target_data_path)
    
    item_df.to_csv(item_file)


def save_user_data(user_df, target_data_path):
    user_file = os.path.join(target_data_path, "user.csv")

    if not os.path.exists(target_data_path):
        os.makedirs(target_data_path)
    
    user_df.to_csv(user_file)

def save_stat_data(rating_df, target_data_path):
    stat_file = os.path.join(target_data_path, "data_stat.json")
    if not os.path.exists(target_data_path):
        os.makedirs(target_data_path)
    data = {
        "num_user": int(rating_df["uid"].max()),
        "num_item": int(rating_df["iid"].max())
    }
    with open(stat_file, "w") as f:
        f.write(json.dumps(data, indent=2))


def save_llmcode_hist(rating_df, target_data_path, max_len=20):
    
    has_timestamp = "timestamp" in rating_df.columns

    has_label = "label" in rating_df.columns

    def collect_data(data, u_hist=None, max_len=10):
        data = data.copy()
        u_hist = u_hist if u_hist else defaultdict(list)
        hist_id_list = []
        hist_rating_list = []
        hist_label_list = []
        if has_timestamp:
            data = data.sort_values(['timestamp', 'uid', 'iid'])

        for row in tqdm(data.to_dict('records'), total=len(data)):
            hist = u_hist[row['uid']][-max_len:]
            hist_id = [h[0] for h in hist]
            if has_label:
                hist_rating = [h[2] for h in hist]
                hist_label = [h[1] for h in hist]
            # else:
            #     hist_rating = [1] * len(hist)

            hist_id_list.append(hist_id)
            hist_rating_list.append(hist_rating)
            hist_label_list.append(hist_label)


            # ## 历史记录只保留喜欢的
            # if only_prefer and row['label'] == 0:
            #     continue
            # else:
            if has_label:
                u_hist[row['uid']].append((row['iid'], row['label'], row['rating']))
            else:
                u_hist[row['uid']].append((row['iid']))
            
        data['hist_id'] = hist_id_list
        if has_label:
            data['hist_rating'] = hist_rating_list
            data['hist_label'] = hist_label_list
        data['his_len'] = data['hist_id'].map(len)
        keys = ['uid', 'iid', 'his_len', 'hist_id',  'rating']
        if has_label:
            keys += ['label', "hist_rating", "hist_label"]
        if has_timestamp:
            keys += ["timestamp"]
        df = data[keys]

        return df.reset_index(drop=True), u_hist

    rating_data, _ = collect_data(rating_df, max_len=max_len)
    
    # Filter out history sequences with fewer than {max_len} items before splitting the dataset
    print(f"Before filtering: {len(rating_data)} records")
    rating_data = rating_data[rating_data['his_len'] >= max_len]
    print(f"After filtering (his_len >= {max_len}): {len(rating_data)} records")
    
    if len(rating_data) == 0:
        print(f"Warning: No data left after filtering for his_len >= {max_len}")
        return


    train_data, val_data, test_data = partition(rating_data)

    def save_file(data, filename, sample_num=-1):
        if sample_num > 0:
            # Note: We already filtered for his_len >= 10, but keeping this for backward compatibility
            # with different sample_num values
            if len(data) > sample_num:
                data = data.sample(sample_num, random_state=RANDOM_SEED)
        data.to_csv(filename, index=False)

    save_file(train_data, os.path.join(target_data_path, "train.csv"))
    save_file(train_data, os.path.join(target_data_path, "train_10000.csv"), sample_num=10000)
    save_file(val_data, os.path.join(target_data_path, "valid.csv"))
    save_file(val_data, os.path.join(target_data_path, "valid_5000.csv"), sample_num=5000)
    save_file(test_data, os.path.join(target_data_path, "test.csv"))
    save_file(test_data, os.path.join(target_data_path, "test_5000.csv"), sample_num=5000)
def print_stat(rating_df, filtered=False):
    print(("filtered" if filtered else "#") + " user:" + str(len(rating_df['uid'].unique())))
    print(("filtered" if filtered else "#") + " item:" + str(len(rating_df['iid'].unique())))
    print(("filtered" if filtered else "#") + " interaction is:" + str(len(rating_df)))

def filter_Ncore(rating_df, k_core=5, max_len=10):
    _filtered_rating_df = rating_df
    
    # _filtered_rating_df['iid_cnt'] = _filtered_rating_df['iid'].map(_filtered_rating_df.value_counts("iid"))
    # _filtered_rating_df['uid_cnt'] = _filtered_rating_df['uid'].map(_filtered_rating_df.value_counts("uid"))
    # _filtered_rating_df = _filtered_rating_df[_filtered_rating_df['uid_cnt'] >= k_core]
    # return _filtered_rating_df
    while True:
        _filtered_rating_df['iid_cnt'] = _filtered_rating_df['iid'].map(_filtered_rating_df.value_counts("iid"))
        _filtered_rating_df['uid_cnt'] = _filtered_rating_df['uid'].map(_filtered_rating_df.value_counts("uid"))
        old_len = len(_filtered_rating_df)
        _filtered_rating_df = _filtered_rating_df[(_filtered_rating_df['iid_cnt'] >= k_core) & (_filtered_rating_df['uid_cnt'] >= max(max_len, k_core))].reset_index(drop=True)
        if len(_filtered_rating_df) == old_len:
            break
    return _filtered_rating_df


def remap(rating_df, start_from_zero=False):
    u_dict = {}
    u_idx = 0 if start_from_zero else 1
    i_dict = {}
    i_idx = 0 if start_from_zero else 1
    for row in rating_df.to_dict('records'):
        if row['uid'] not in u_dict:
            u_dict[row['uid']] = u_idx
            u_idx = u_idx + 1

        if row['iid'] not in i_dict:
            i_dict[row['iid']] = i_idx
            i_idx = i_idx + 1
    return u_dict, i_dict