import pandas as pd
import os
import json
from tqdm import tqdm
import gzip
import numpy as np
import argparse
from collections import defaultdict
import jsonlines
import random
from preprocess_utils import *
from string import ascii_letters, digits, punctuation, whitespace
import html

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
pd.options.mode.chained_assignment = None  # default='warn'

raw_data_path = "../../rawdata/amazon2018"

def filter_title(x):
    x = html.unescape(x)
    x = x.replace("“", "\"")
    x = x.replace("”", "\"")
    x = x.replace("‘", "'")
    x = x.replace("’", "'")
    x = x.replace("–", "-")
    x = x.replace("\n", " ")
    x = x.replace("\r", " ")
    x = x.replace("…", "")
    x = x.replace("‚", ",")
    x = x.replace("´", "'")
    x = x.replace("&ndash;", "-")
    x = x.replace("&lt;", "<")
    x = x.replace("&gt;", ">")
    x = x.replace("&amp;", "&")
    x = x.replace("&quot;", "\"")
    x = x.replace("&nbsp;", " ")
    x = x.replace("&copy;", "©")
    x = x.replace("″", "\"")
    x = x.replace("【", "[")
    x = x.replace("】", "]")
    x = x.replace("—", "-")
    x = x.replace("−", "-")
    for ci in [127, 174, 160, 170, 8482, 188, 162, 189, 8594, 169, 235, 168, 957, 12288, 8222, 179, 190, 173, 186, 8225]:
        x = x.replace(chr(ci), " ")
    while "  " in x:
        x = x.replace("  ", " ")
    x = x.strip()
    if 'abcdefg' in x.lower():
        # print(x) 
        x = 'unknown title'
    for letter in x:
        if letter not in ascii_letters + digits + punctuation + whitespace:
            return ascii_letters + digits + punctuation + whitespace # Delete Flag
    if len(set(x) & set(ascii_letters)) == 0 or len(x) < 3:
        return ascii_letters + digits + punctuation + whitespace # Delete Flag
    if len(x) > 150:
        x = x[:150]
        x = " ".join(x.strip().split(" ")[:-1])
        while x[-1] not in ascii_letters + digits:
            x = x[:-1]
    return x


def convert_category(x):
    if isinstance(x, list):
        x = x[1:]
        x = [x[0].strip()] if len(x) else ['unknown category']
    else:
        x = ['unknown category']
    return x


def convert_brand(x):
    if isinstance(x, list):
        x = x[0]
    x = html.unescape(x)
    x = x.replace("“", "\"")
    x = x.replace("”", "\"")
    x = x.replace("‘", "'")
    x = x.replace("’", "'")
    x = x.replace("–", "-")
    x = x.replace("\n", " ")
    x = x.replace("\r", " ")
    x = x.replace("…", "")
    x = x.replace("‚", ",")
    x = x.replace("´", "'")
    x = x.replace("&ndash;", "-")
    x = x.replace("&lt;", "<")
    x = x.replace("&gt;", ">")
    x = x.replace("&amp;", "&")
    x = x.replace("&quot;", "\"")
    x = x.replace("&nbsp;", " ")
    x = x.replace("&copy;", "©")
    x = x.replace("″", "\"")
    x = x.replace("【", "[")
    x = x.replace("】", "]")
    x = x.replace("—", "-")
    x = x.replace("−", "-")
    for ci in [127, 174, 160, 170, 8482, 188, 162, 189, 8594, 169, 235, 168, 957, 12288, 8222, 179, 190, 173, 186, 8225]:
        x = x.replace(chr(ci), " ")
    while "  " in x:
        x = x.replace("  ", " ")
    x = x.strip()
    if 'abcdefg' in x.lower():
        # print(x)
        x = 'unknown brand'
        return x
    for letter in x:
        if letter not in ascii_letters + digits + punctuation + whitespace:
            return ascii_letters + digits + punctuation + whitespace # Delete Flag
    if len(set(x) & set(ascii_letters)) == 0 or len(x) < 3:
        return ascii_letters + digits + punctuation + whitespace # Delete Flag
    if len(x) > 150:
        x = x[:150]
        x = " ".join(x.strip().split(" ")[:-1])
        while x[-1] not in ascii_letters + digits:
            x = x[:-1]
    return x

parser = argparse.ArgumentParser(description='llm_seq')
# GPU配置
parser.add_argument('--max_len', type=int, default=10)
parser.add_argument("--dataset", type=str, default="Amazon_Fashion")
parser.add_argument("--start_from_one", default=False, action="store_true")
parser.add_argument("--duration", default=-1, type=int)
# 训练配置



args = parser.parse_args() 
only_prefer = True #args.only_prefer
max_len = args.max_len
start_from_one =  True#args.start_from_one
duration = args.duration
leave_one_out = False
k_core = 5
target_path = "../data/"


dataset = args.dataset

target_data_path = os.path.join(target_path, dataset)

os.makedirs(target_data_path, exist_ok=True)
filename = dataset
filter_by_desc = False
rating_num = 100000000
detailKeys = {
}
duration = args.duration
if dataset == "Toys_and_Games":
    duration = 48
    detailKeys = {
        "brand": ["Brand", "brand"],
        # "style": ["Style"],
    }
elif dataset == "Sports_and_Outdoors":
    duration = 48
    detailKeys = {
        "brand": ["Brand", "brand"],
        "style": ["Style"],
        "category": ["Category", "category", "Categories", "categories", "cate", "cates"],
    }
elif dataset == "Office_Products":
    duration = -1
meta_file = os.path.join(raw_data_path, "meta_" + filename + ".json.gz")

rating_file = os.path.join(raw_data_path, filename + ".json.gz")

df_data = []
rawdata = []
keys = list(detailKeys.keys())

item_mapping = dict()
with gzip.open(meta_file) as f:
    for l in tqdm(f.readlines()):
        l = l.decode()
        line = json.loads(l)
        if "title" not in line or not line['title'] or line['title'] in ['unknown', "None"]:
            continue
        title = line['title'][:50]
        extra_details = []
        for k in keys:
            current_details = []
            for detailKey in detailKeys[k]:
                if detailKey in line:
                    if type(line[detailKey]) == str:
                        current_details.append(line[detailKey])
                    else:
                        current_details.extend(line[detailKey])
            if "details" in line and line['details'] is not None:
                for detailKey in detailKeys[k]:
                    if detailKey in line['details']:
                        if type(line['details'][detailKey]) == str:
                            current_details = [line['details'][detailKey]]
                        else:
                            current_details = line['details'][detailKey]
                        ### 多个Key只要命中一个就行
                        break
            extra_details.append(current_details)
        asin = line['asin'] if "asin" in line and line['asin'] is not None else line['parent_asin']
        mapping_asin = asin
        if title in item_mapping:
            mapping_asin = item_mapping[title]
        else:
            item_mapping[title] = asin
        # rawdata.append(line)
        df_data.append([asin, title, mapping_asin] + extra_details)

item_meta_df = pd.DataFrame(df_data, columns = ['iid', 'title', "mapping_iid"] + keys)
item_meta_df['title'] = item_meta_df['title'].apply(filter_title)

if 'category' in item_meta_df.columns:
    item_meta_df['category'] = item_meta_df['category'].apply(convert_category)

if 'brand' in item_meta_df.columns:
    item_meta_df['brand'] = item_meta_df['brand'].apply(convert_brand)
    item_meta_df['brand_exists'] = item_meta_df['brand'].apply(lambda x: "abcde" not in x.lower())
    item_meta_df = item_meta_df[item_meta_df['brand_exists']]
    item_meta_df = item_meta_df.drop(columns=['brand_exists'])

item_meta_df = item_meta_df.drop_duplicates("iid").reset_index(drop=True)

if filter_by_desc:
    print("before filtering, the number of the item is:" + str(len(item_meta_df)))
    item_meta_df = item_meta_df[item_meta_df['description'].map(len) > 0].reset_index(drop=True)

print("current number of the item is:" + str(len(item_meta_df)))
item_meta_df_dict = item_meta_df.reset_index().set_index('iid')

## mapping the duplicate title
rating_data = []
i = 0
u_dict = {}
cache_file = os.path.join(target_path, dataset + "_rating_cache.csv")

if os.path.exists(cache_file):
    rating_df = pd.read_csv(cache_file, )
else:
    with gzip.open(rating_file) as f:
        for l in tqdm(f.readlines()[:rating_num]):
            l = l.decode() 
            line = json.loads(l)
            
            item = None
            if "asin" in line and line['asin'] is not None:
                item = line['asin']
            else:
                item = line['parent_asin']

            if item not in item_meta_df_dict.index:
                continue
            
            item_obj = item_meta_df_dict.loc[item]

            rating = line['overall']
            userid = line['user_id'] if "user_id" in line else line['reviewerID']
            timestamp = line['timestamp'] if "timestamp" in line else line["unixReviewTime"]

            rating_data.append([userid, item, item_obj['mapping_iid'], rating, timestamp]) 
    rating_df = pd.DataFrame(rating_data, columns=['uid', 'iid', 'mapping_iid', 'rating', 'timestamp'])
    # rating_df['iid'] = rating_df["mapping_iid"]
    rating_df.to_csv(cache_file, index=False)

if duration >= 0:
    raw_rating_df = rating_df
    timestamp = rating_df['timestamp'].max() - duration *30*24*60*60
    rating_df = rating_df[rating_df['timestamp'] > timestamp]
rating_df = rating_df[rating_df['iid'].isin(item_meta_df['iid'].unique())]

print_stat(rating_df)
filtered_rating_df = filter_Ncore(rating_df, k_core, max_len=max_len) 
    
### map string to id
print_stat(filtered_rating_df, filtered=True)

filtered_rating_df = filtered_rating_df.sort_values(['timestamp', "uid", "iid"]).reset_index(drop=True)
filtered_item_df = item_meta_df[item_meta_df['iid'].isin(filtered_rating_df['iid'].unique())]
filtered_item_df = filtered_item_df.rename(columns={'iid': 'rawid'})


u_dict, i_dict = remap(filtered_rating_df)


filtered_item_df['iid'] = filtered_item_df['rawid'].map(i_dict)
filtered_item_df = filtered_item_df.sort_values("iid").reset_index(drop=True)

# Save original user IDs before mapping
filtered_user_df = pd.DataFrame(filtered_rating_df['uid'].unique(), columns=["rawid"])

# Map item and user IDs
filtered_rating_df['iid'] = filtered_rating_df['iid'].map(i_dict)
filtered_rating_df['uid'] = filtered_rating_df['uid'].map(u_dict)
filtered_rating_df['label'] = filtered_rating_df.apply(lambda row: 1 if row['rating'] > 4 else 0, axis=1)

filtered_rating_df = filtered_rating_df.drop(columns=['iid_cnt', 'uid_cnt'])

# Create filtered_user_df with both original and new user IDs
filtered_user_df['uid'] = filtered_user_df['rawid'].map(u_dict)
filtered_user_df = filtered_user_df.sort_values("uid").reset_index(drop=True)
print("filtered item duplicated number:" + str(filtered_item_df.duplicated("title").sum()))

print("saving items")

save_item_data(filtered_item_df, target_data_path)
print("saving users")
save_user_data(filtered_user_df, target_data_path)

save_stat_data(filtered_rating_df, target_data_path)
print("saving llmcode dataset")
save_llmcode_hist(filtered_rating_df, target_data_path, max_len=max_len)
print("all dataset complete")   
