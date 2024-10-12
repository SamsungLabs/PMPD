# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py

from dataclasses import dataclass, field
from tqdm import tqdm
import pathlib
import os
from typing import Dict, Optional, Sequence
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from transformers import AutoConfig
import argparse

from torch.nn import functional as F
import os
from hyperopt import STATUS_OK

from pmpd import Scheduler

def eval(config, train_dataset, precisions, classifiers_dir=None):
    # Scheduler
    precision_switch_points = set()
    for p in sorted(precisions, reverse=True):
        if p-1 in precisions:
            precision_switch_points.add((p, p-1))
    scheduler = Scheduler['kv_cache'](precisions, 
                                      precision_switch_points, 
                                      dim=config.hidden_size // config.num_attention_heads,
                                      num_heads=config.num_key_value_heads,
                                      save_dir=classifiers_dir)
                                

    # Load data
    train_dataset_index = len(train_dataset) * 9 // 10 
    for ((high_bit, low_bit), classifier) in scheduler.classifiers.items():
        classifier.cuda()
        classifier.eval()
        
        train_loss = 0
        with torch.no_grad():
            for i in range(train_dataset_index):
                inputs = train_dataset[i]
                past_key_values = inputs['input_ids']
                labels = inputs['labels']
                past_key = past_key_values[(high_bit, low_bit)][0]
                past_value = past_key_values[(high_bit, low_bit)][1]
                pred = classifier(past_key.cuda(), past_value.cuda())[0]
                label = torch.tensor([labels[(high_bit, low_bit)]], dtype=pred.dtype).cuda()
                train_loss += F.binary_cross_entropy(pred, label)
        train_loss = train_loss / train_dataset_index
        print(f"Training Loss {train_loss.cpu().item()}", flush=True)
        
        eval_loss = 0
        with torch.no_grad():
            for i in range(train_dataset_index, len(train_dataset)):
                inputs = train_dataset[i]
                past_key_values = inputs['input_ids']
                labels = inputs['labels']
                past_key = past_key_values[(high_bit, low_bit)][0]
                past_value = past_key_values[(high_bit, low_bit)][1]
                pred = classifier(past_key.cuda(), past_value.cuda())[0]
                label = torch.tensor([labels[(high_bit, low_bit)]], dtype=pred.dtype).cuda()
                eval_loss += F.binary_cross_entropy(pred, label)
        eval_loss = eval_loss / (len(train_dataset) - train_dataset_index)
        print(f"Validation Loss {eval_loss.cpu().item()}", flush=True)


    return {'loss': eval_loss.cpu().item(), 'status': STATUS_OK, 'model': scheduler.classifiers}

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_path', type=str, required=True)
    argparser.add_argument('--data_path', type=str, required=True)
    argparser.add_argument('--precisions', type=str, default='2,3,4')
    argparser.add_argument('--classifiers_path', type=str, default=None)
    
    args = argparser.parse_args()
    
    config = AutoConfig.from_pretrained(args.model_path)
    train_dataset = torch.load(args.data_path)
    precision_list = [int(p) for p in args.precisions.split(',')]
    
    eval(config, train_dataset, precision_list, args.classifiers_path)

    