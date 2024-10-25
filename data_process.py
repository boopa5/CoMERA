#process MNLI
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from datasets import load_dataset
import argparse
import time

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.utils.data

batch=128

dataset = load_dataset("glue",'mnli')
# dataset = train_iter

from transformers import DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def truncate(id1,id2):
    while len(id1)+len(id2)>128-3:
        if len(id1)>=len(id2):
            id1 = id1[:-1]
        else:
            id2 = id2[:-1]
    
    return id1,id2
    
def tokenize_function(examples):
    id1 = tokenizer(examples['premise'])['input_ids'][1:-1]
    id2 = tokenizer(examples['hypothesis'])['input_ids'][1:-1]
    
    id1,id2 = truncate(id1,id2)
    
    id1 = [101] + id1 + [102]
    id2 = id2 + [102]
    
    token_ids = [0]*len(id1) + [1]*len(id2)
    
    ids = id1+id2
    attn = [1]*len(id1+id2)
    
#     print(len(ids),len(attn),len(token_ids))
    
    
#     out = tokenizer(examples['premise'],examples['hypothesis'],return_token_type_ids=True)
#     ids,attn,token_ids = out['input_ids'],out['attention_mask'],out['token_type_ids']
    return {'ids':ids,'attn':attn,'token_ids':token_ids}

dataset = dataset.filter(lambda x:len(x['premise'])>0 and len(x['hypothesis'])>0)
tokenized_datasets = dataset.map(tokenize_function, num_proc=32, remove_columns=["premise","hypothesis","idx"])

# new_data = tokenized_datasets.filter(lambda x: len(x['input_ids'])>0 )

tokenized_datasets.save_to_disk('./datasets/MNLI_len128/')
