import sys
import io, os
import numpy as np
from typing import Optional, Union, List, Dict, Tuple
import logging
import argparse
from prettytable import PrettyTable
import transformers
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from loguru import logger


from torch.nn import CosineSimilarity
from tqdm import tqdm

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# file_path = args.test_file
# test_dataset = pd.read_csv(file_path, sep=',' if 'csv' in file_path else None)

def load_data(file_path):
    return pd.read_csv(file_path, sep=',' if 'csv' in file_path else None)

def get_features(df, tokenizer):
    test_feature_list = []
    for i in range(len(df.index)):
        encoded_sent1 = tokenizer(df['sent0'][i], truncation=True, padding='max_length', return_tensors="pt")
        encoded_sent2 = tokenizer(df['sent1'][i], truncation=True, padding='max_length', return_tensors="pt")
        test_feature_list.append((encoded_sent1, encoded_sent2))
    return test_feature_list

def evaluate(model, dataloader, device):
    model.eval()
    cos = CosineSimilarity(dim=-1)
    
    # sim_tensor = torch.tensor([], device=device)
    # label_array = np.array([])
    with torch.no_grad():
        sim_scores=[]
        for sent1, sent2 in tqdm(dataloader):
            # print(sent1)
            # print('-----------------------')
            # print(sent2)
            sent1_input_ids = sent1.get('input_ids').squeeze(1).to(device)
            sent1_attention_mask = sent1.get('attention_mask').squeeze(1).to(device)
            sent1_token_type_ids = sent2.get('token_type_ids').squeeze(1).to(device)
            sent1_pred = model(sent1_input_ids, sent1_attention_mask, sent1_token_type_ids).last_hidden_state[:, 0]
            # print('input ids:', sent1_input_ids.size())
            # print('pred', sent1_pred.size())
            # sent2
            sent2_input_ids = sent2.get('input_ids').squeeze(1).to(device)
            sent2_attention_mask = sent2.get('attention_mask').squeeze(1).to(device)
            sent2_token_type_ids = sent2.get('token_type_ids').squeeze(1).to(device)
            sent2_pred = model(sent2_input_ids, sent2_attention_mask, sent2_token_type_ids).last_hidden_state[:, 0]
            
            sim = cos(sent1_pred, sent2_pred).item()
            # print(f'sim score is {sim}')
            sim_scores.append(sim)
    
    return sim_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, 
            default="./result/",
            help="Transformers' model name or path")
    parser.add_argument("--pooler", type=str, 
            choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], 
            default='cls', 
            help="Which pooler to use")
    parser.add_argument("--test_file", type=str, 
            default="./data/COMPLAINTS_RECEIVED_2020_2023/com_data/test.csv",
            help="")

    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load transformers' model checkpoint
    model = AutoModel.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = model.to(device)
    
    test_df = load_data(args.test_file)
    test_feature_list = get_features(test_df, tokenizer=tokenizer)
    test_dataset = TestDataset(test_feature_list, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset)

    
    sim_scores = evaluate(model, dataloader=test_dataloader, device=device)
    logger.info('sim scores:{}'.format(sim_scores))

if __name__ == "__main__":
    main()