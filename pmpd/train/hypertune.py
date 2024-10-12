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
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import Trainer, BitsAndBytesConfig, AutoConfig
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import os
import argparse

from pmpd import Scheduler

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from functools import partial
from train_scheduler import SupervisedDataset, train

# Best hyperparameters: {'dropout_rate': 0.6415047644528229, 'gradient_accumulation_steps': 2, 'learning_rate': 5.915599068579025e-05, 'num_train_epochs': 2, 'weight_decay': 0.011158596518451974}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--precisions', type=str, default='2,3,4', help='Precisions to use')
    parser.add_argument('--all_layers', action='store_true', default=False)
    parser.add_argument('--postfix_name', type=str, help='postfix string to avoid overwritting existing scheduler', default='')
    args = parser.parse_args()
    
    search_space = {
        'learning_rate': hp.uniform('learning_rate', 1e-5, 1e-4),
        'num_train_epochs': hp.choice('num_train_epochs', [3, 5, 10]),
        'gradient_accumulation_steps': hp.choice('gradient_accumulation_steps', [32, 64, 128]),
        # 'weight_decay': hp.uniform('weight_decay', 0.01, 0.02),
        'weight_decay': hp.uniform('weight_decay', 0.01, 0.02),
        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.3), 
        # 'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.3),
    }
    
    config = AutoConfig.from_pretrained(args.model_path)
    dataset = torch.load(args.data_path)

    precisions = [int(p) for p in args.precisions.split(',')]
    objective = partial(train, config=config, dataset=dataset, precisions=precisions, all_layers=args.all_layers)

    trials = Trials()
    best = fmin(fn=objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=3,
                trials=trials)

    print(f"Best hyperparameters: {best}")
    
    # Save the best model
    best_trial = trials.best_trial
    best_model = best_trial['result']['model']
    model_name = args.data_path.split("/")[-1].split("train_dataset_")[-1].split(".pt")[0]
    output_dir = f"test/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    for (high_bit, low_bit), classifier in best_model.items():
        torch.save(classifier.state_dict(), f"{output_dir}/classifier_{high_bit}_{low_bit}{args.postfix_name}.pt")