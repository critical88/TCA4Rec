import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import json


class TrainCollater:
    def __init__(self,
                 args=None,
                 tokenizer=None,
                 train=False,
                 ):
        self.args = args
        self.tokenizer = tokenizer
        self.train = train

    def __call__(self, batch):
        input_ids = []
        attention_mask = []
        labels = []
        ## position_mask is the token_type_ids
        pad_batch = self.tokenizer.pad(batch, return_tensors=None)
        rt_batch = pad_batch
        rt_batch = {
            "input_ids": torch.LongTensor(pad_batch['input_ids']),
            "attention_mask": torch.LongTensor(pad_batch['attention_mask']),
            "title_idx": torch.LongTensor(pad_batch['title_idx']),
            'data_idx': torch.LongTensor(pad_batch['data_idx'])
        }
        if "user_idx" in pad_batch:
            rt_batch["user_idx"] = torch.LongTensor(pad_batch['user_idx'])
        if "hist_id" in pad_batch:
            rt_batch["hist_id"] = torch.LongTensor(pad_batch['hist_id'])
        
        return rt_batch
class LLM4RecDataset(Dataset):
    def __init__(self, args, n_items, tokenizer, data, item_df, similar_items=None, train=False):
        self.args = args
        self.tokenizer = tokenizer
        self.data = data
        self.n_items = n_items
        self.similar_items = similar_items
        self.max_seq_length = self.args.max_seq_length
        self.train = train
        self.itemid2title = item_df.set_index("iid")['title'].to_dict()
        self.item_pad_id = 0
        idx = 128
        if idx < len(data):
            self.transform(idx)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.transform(idx)
    def get_prompt(self, hist_id):
        dataset = self.args.dataset 
        domain = "item"
        if dataset == "Office_Products":
            domain = "office products"
        elif dataset == "Toys_and_Games":
            domain ="games"
        elif dataset == "Sports_and_Outdoors":
            domain = "sports"
        hist_title = [self.itemid2title[h] for h in hist_id if h != 0]
        instruction = f"""
###Instruction:
Given the user interaction history, please recommendate the {domain} that user is most likely to view.

###Input:
User Interaction Histories: [HistoryHere].

Then the movie that user is most likely to watch is:
###Response:
"""
        if self.args.use_feature:
            inputs = instruction.replace("[HistoryHere]", ", ".join(f'"{h}"[ItemEmb]' for h in hist_title)).strip()
            inputs = "[UserEmb]" + inputs
        else:
            inputs = instruction.replace("[HistoryHere]", ", ".join(f'"{h}"' for h in hist_title)).strip()

        return inputs


    # dataset中对于每条数据的前置处理
    def transform(self, idx):
        data = self.data.iloc[idx]
        
        max_item_len = self.args.max_item_len
        
        item_id = data["iid"]
        hist_id = data['hist_id']
        user_id = data['uid']
        item_title = self.itemid2title[item_id]
        inputs = self.get_prompt(hist_id)
        
        output = item_title

        inputs_ids = self.tokenizer(inputs)['input_ids']
        output_ids = []
        if self.train:
            output_ids = self.tokenizer(output)['input_ids']
            if self.tokenizer.bos_token_id is not None and output_ids[0] != self.tokenizer.bos_token_id:
                output_ids = [self.tokenizer.bos_token_id] + output_ids
        elif self.tokenizer.bos_token_id is not None:
            output_ids = [self.tokenizer.bos_token_id]


        final_input_ids = inputs_ids + output_ids
        attention_mask = [1] * len(final_input_ids)

        feature = {}
        feature['input_ids'] = final_input_ids
        feature['title_idx'] = item_id
        feature['user_idx'] = user_id
        feature['data_idx'] = idx

        feature['hist_id'] = [self.item_pad_id] * (max_item_len -len(hist_id)) + hist_id

        feature['attention_mask'] = attention_mask

        return feature

class LLM4RecDataModule(LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.is_evalution = not not (args.ckpt_path) 
        self.tokenizer = tokenizer
        self.max_seq_length = self.args.max_seq_length
        datapath = os.path.join(args.data_path, args.dataset)
        item_data_path = os.path.join(datapath, "item.csv")
        item_df = pd.read_csv(item_data_path)
        item_df['title'] = item_df.apply(lambda x: x['title'][:100], axis=1)
        self.item_df = item_df

        self.rawitemid2title = item_df.set_index("iid")['title'].to_dict()
        self.generate_item_token_ids()
        train_data = pd.read_csv(os.path.join(datapath, "train_10000.csv"))
        val_data = pd.read_csv(os.path.join(datapath, "valid_5000.csv"))
        test_data = pd.read_csv(os.path.join(datapath, "test_5000.csv"))

        with open(os.path.join(datapath, "data_stat.json")) as f:
            data_stat = json.load(f)
        self.n_user = data_stat['num_user']
        self.n_item = data_stat['num_item']
        train_data['hist_id'] = train_data['hist_id'].apply(eval)
        val_data['hist_id'] = val_data['hist_id'].apply(eval)
        test_data['hist_id'] = test_data['hist_id'].apply(eval)
        
        # train_data = train_data[:10]
        # val_data = val_data[:10]
        # test_data = test_data[:10]
        self.train_dataset = LLM4RecDataset(args, self.n_item, self.tokenizer, train_data, item_df, train=True)
        self.val_dataset = LLM4RecDataset(args, self.n_item, self.tokenizer, val_data, item_df)
        self.test_dataset = LLM4RecDataset(args, self.n_item, self.tokenizer, test_data, item_df)

        collater = TrainCollater(self.args, self.tokenizer, train=True)
        collater.__call__([self.train_dataset[0], self.train_dataset[1]])

    def generate_item_token_ids(self):
        """
        Generate token ID lists for all item titles and store them in a dictionary.
        """
        self.item_token_ids = {}
        for iid, title in self.item_df[['iid', 'title']].values:
            token_ids = self.tokenizer(title, add_special_tokens=False)['input_ids']
            self.item_token_ids[iid] = token_ids

    def train_dataloader(self):
        if self.is_evalution:
            return None
        return DataLoader(self.train_dataset, batch_size=self.args.micro_batch_size, shuffle=True, num_workers=1, collate_fn=TrainCollater(
                self.args, self.tokenizer, train=True, 
            ))

    def val_dataloader(self):
        
        return DataLoader(self.val_dataset, batch_size=self.args.micro_batch_size, shuffle=False, num_workers=1, collate_fn=TrainCollater(
                self.args, self.tokenizer, train=False, 
            ))
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.micro_batch_size, shuffle=False, num_workers=1, collate_fn=TrainCollater(
                self.args, self.tokenizer, train=False, 
            ))