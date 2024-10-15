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
import random

from torch.nn import functional as F
import os
import evaluate
from hyperopt import STATUS_OK
from sklearn.metrics import precision_recall_curve
from any_precision.modules.AnyPrecisionLinear import AnyPrecisionLinear

from pmpd import Scheduler

def clone_past_key_values(past_key_values, device='cuda'):
    return tuple(
        tuple(past_key_value.clone().to(device) for past_key_value in layer)
        for layer in past_key_values
    )

def set_hook(layer_idx, acts):
    if layer_idx not in acts:
        acts[layer_idx] = {} 
    def hook(m, inp, oup):
        inp = inp[0]

        acts[layer_idx]['mean'] = inp.mean().cpu()
        acts[layer_idx]['var'] = inp.var(correction=1).cpu()
        acts[layer_idx]['acts'] = inp.detach().cpu()

    return hook

class SupervisedActDatasetv3(Dataset):
    def __init__(self, dataset, model, precision_switch_points, size=None):
        super(SupervisedActDatasetv3, self).__init__()
        label_dict = {}
        input_dict = {}
        input_all_layers_dict = {}
        rouge = evaluate.load('rouge')
        size = size if size is not None else len(dataset)

        hooks = {}
        for (high_bit, low_bit) in precision_switch_points:
            labels = []
            inputs = [] # inputs of last layer
            inputs_all_layers = [] # mean and var of all_layers
            for data in tqdm(dataset[:size]):
                high_prec_steps = [0, 255 // 3, 255 // 3 * 2]
                mix_outputs = [None] * len(high_prec_steps)
                mix_input_ids = [None] * len(high_prec_steps)
                with torch.inference_mode():
                    input_ids = data['input_ids'].cuda()
                    input_len = input_ids.shape[1]

                    # Hooking to collect acts
                    acts = {}
                    linear_layer_idx = 0
                    for n, m in model.named_modules():
                        if isinstance(m, (nn.Linear, AnyPrecisionLinear)):
                            hooks[linear_layer_idx] = m.register_forward_hook(set_hook(linear_layer_idx, acts))
                            linear_layer_idx += 1
                    
                    outputs = model.model(input_ids, precision=high_bit, past_key_values=None, use_cache=True)  

                    for h in hooks.values():
                        h.remove()

                    # save inputs of last layer
                    inputs.append(acts[list(acts.keys())[-1]]['acts'])

                    _temp = []
                    for layer_acts in acts.values():
                        _temp.append(layer_acts['mean'])
                        _temp.append(layer_acts['var'])
                    
                    inputs_all_layers.append(torch.stack(_temp))

                    # print('1', inputs[-1][0].shape)
                    new_token = 0

                    while new_token < 256:
                        input_id = outputs['logits'][:, -1:].argmax(dim=-1)
                        outputs = model.model(input_id, 
                                                precision=high_bit, 
                                                use_cache=True, 
                                                past_key_values=outputs['past_key_values'])
                        input_ids = torch.cat([input_ids, input_id], dim=-1)
                        for i, step in enumerate(high_prec_steps):
                            if new_token == step:
                                mix_outputs[i] = {'logits': outputs['logits'].clone(), 
                                                'past_key_values': clone_past_key_values(outputs['past_key_values'])}
                                mix_input_ids[i] = input_ids.clone()
                            elif new_token > step:
                                input_id = mix_outputs[i]['logits'][:, -1:].argmax(dim=-1)
                                mix_outputs[i] = model.model(input_id, 
                                                            precision=low_bit, 
                                                            use_cache=True, 
                                                            past_key_values=mix_outputs[i]['past_key_values'])
                                mix_input_ids[i] = torch.cat([mix_input_ids[i], input_id], dim=-1)
                                
                        
                        new_token += 1
                        if model.tokenizer.eos_token_id in input_ids[0, :].tolist():
                            break
                        torch.cuda.empty_cache() 
                high_bit_tokens = model.tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=True)
                # print('input tokens:\n', model.tokenizer.decode(input_ids[0, :], skip_special_tokens=True))
                # print('high bit tokens:\n', high_bit_tokens)
                # print('fp16 output:\n', data['fp16_output'])
                high_bit_rouge = rouge.compute(predictions=[high_bit_tokens], references=[data['fp16_output']])['rougeL']
                # print(f'high bit rouge: {high_bit_rouge}')
                step_candidates = [len(high_prec_steps)]
                for i, mix_outputs_i in enumerate(mix_outputs):
                    if mix_outputs_i is None:
                        continue
                    mix_bit_tokens = model.tokenizer.decode(mix_input_ids[i][0, input_len:], skip_special_tokens=True)
                    # print(f'mix bit tokens {i}:\n', mix_bit_tokens)
                    mix_bit_rouge = rouge.compute(predictions=[mix_bit_tokens], references=[data['fp16_output']])['rougeL']
                    # print(f'mix bit rouge {i}: {mix_bit_rouge}')
                    if round(high_bit_rouge, 3) <= round(mix_bit_rouge, 3):
                        step_candidates.append(i)
                labels.append(min(step_candidates))
                print('labels', labels)
                # print('2', inputs[-1][0].shape)
            label_dict[(high_bit, low_bit)] = labels
            input_dict[(high_bit, low_bit)] = inputs
            input_all_layers_dict[(high_bit, low_bit)] = inputs_all_layers
            torch.cuda.empty_cache() 
    

        for (high_bit, low_bit) in precision_switch_points:
            count_dict = defaultdict(int)
            for label in label_dict[(high_bit, low_bit)]:
                count_dict[label] += 1
            print(f"Count for precision {high_bit}, {low_bit}: {count_dict}")
        self.input_dict = input_dict
        self.input_all_layers_dict = input_all_layers_dict
        self.label_dict = label_dict
        self.precisions = precision_switch_points[0]

    def set_precision(self, precision):
        self.precisions = precision

    def __len__(self):
        return len(self.input_dict[self.precisions])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_dict[self.precisions][i],
            input_all_layers=self.input_all_layers_dict[self.precisions][i],
            labels=self.label_dict[self.precisions][i],
        )

class SupervisedActDatasetv2(Dataset):
    def __init__(self, dataset, model, precision_switch_points, size=None):
        super(SupervisedActDatasetv2, self).__init__()
        label_dict = {}
        input_dict = {}
        input_all_layers_dict = {}
        rouge = evaluate.load('rouge')
        size = size if size is not None else len(dataset)

        hooks = {}
        for (high_bit, low_bit) in precision_switch_points:
            labels = []
            inputs = [] # inputs of last layer
            inputs_all_layers = [] # mean and var of all_layers
            for data in tqdm(dataset[:size]):
                high_prec_steps = [0, 255 // 3, 255 // 3 * 2]
                mix_outputs = [None] * len(high_prec_steps)
                mix_input_ids = [None] * len(high_prec_steps)
                with torch.inference_mode():
                    input_ids = data['input_ids'].cuda()
                    input_len = input_ids.shape[1]

                    outputs = model.model(input_ids, precision=high_bit, past_key_values=None, use_cache=True)  

                    # print('1', inputs[-1][0].shape)
                    new_token = 0

                    while new_token < 256:
                        input_id = outputs['logits'][:, -1:].argmax(dim=-1)
                        outputs = model.model(input_id, 
                                                precision=high_bit, 
                                                use_cache=True, 
                                                past_key_values=outputs['past_key_values'])
                        input_ids = torch.cat([input_ids, input_id], dim=-1)
                        for i, step in enumerate(high_prec_steps):
                            if new_token == step:
                                mix_outputs[i] = {'logits': outputs['logits'].clone(), 
                                                'past_key_values': clone_past_key_values(outputs['past_key_values'])}
                                mix_input_ids[i] = input_ids.clone()
                            elif new_token > step:
                                input_id = mix_outputs[i]['logits'][:, -1:].argmax(dim=-1)

                                if new_token == 255 or model.tokenizer.eos_token_id in input_ids[0, :].tolist():
                                    acts = {}
                                    linear_layer_idx = 0
                                    for m in model.modules():
                                        if isinstance(m, (nn.Linear, AnyPrecisionLinear)):
                                            hooks[linear_layer_idx] = m.register_forward_hook(set_hook(linear_layer_idx, acts))
                                            linear_layer_idx += 1
                                mix_outputs[i] = model.model(input_id, 
                                                            precision=low_bit, 
                                                            use_cache=True, 
                                                            past_key_values=mix_outputs[i]['past_key_values'])
                                mix_input_ids[i] = torch.cat([mix_input_ids[i], input_id], dim=-1)

                                if new_token == 255 or model.tokenizer.eos_token_id in input_ids[0, :].tolist():
                                    for h in hooks.values():
                                        h.remove()
                                    mix_outputs[i]['last_layer_acts'] = acts[list(acts.keys())[-1]]['acts']
                                    _temp = []
                                    for layer_acts in acts.values():
                                        _temp.append(layer_acts['mean'])
                                        _temp.append(layer_acts['var'])
                                    
                                    mix_outputs[i]['overall_acts'] = torch.stack(_temp)
                        
                        new_token += 1
                        if model.tokenizer.eos_token_id in input_ids[0, :].tolist():
                            break
                        torch.cuda.empty_cache() 
                high_bit_tokens = model.tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=True)
                high_bit_rouge = rouge.compute(predictions=[high_bit_tokens], references=[data['fp16_output']])['rougeL']
                for i, mix_outputs_i in enumerate(mix_outputs):
                    if mix_outputs_i is None:
                        continue
                    mix_bit_tokens = model.tokenizer.decode(mix_input_ids[i][0, input_len:], skip_special_tokens=True)
                    mix_bit_rouge = rouge.compute(predictions=[mix_bit_tokens], references=[data['fp16_output']])['rougeL']

                    labels.append(round(high_bit_rouge, 2) <= round(mix_bit_rouge, 2))
                    inputs.append(mix_outputs_i['last_layer_acts'])
                    inputs_all_layers.append(mix_outputs_i['overall_acts'])

            label_dict[(high_bit, low_bit)] = labels
            input_dict[(high_bit, low_bit)] = inputs
            input_all_layers_dict[(high_bit, low_bit)] = inputs_all_layers
            torch.cuda.empty_cache() 

        for (high_bit, low_bit) in precision_switch_points:
            count_dict = defaultdict(int)
            for label in label_dict[(high_bit, low_bit)]:
                count_dict[label] += 1
            print(f"Count for precision {high_bit}, {low_bit}: {count_dict}")
        self.input_dict = input_dict
        self.input_all_layers_dict = input_all_layers_dict
        self.label_dict = label_dict
        self.precisions = precision_switch_points[0]

    def set_precision(self, precision):
        self.precisions = precision

    def __len__(self):
        return len(self.input_dict[self.precisions])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_dict[self.precisions][i],
            input_all_layers=self.input_all_layers_dict[self.precisions][i],
            labels=self.label_dict[self.precisions][i],
        )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        dataset (List[Dict[str, torch.Tensor]]): The dataset to fine-tune on.
        model (AnyPrecisionForCausalLM): The model to fine-tune.
        alpha (float): The alpha value for the key value diff threshold.
    """

    def __init__(self, dataset, model, precision_switch_points, size=None):
        super(SupervisedDataset, self).__init__()
        label_dict = {}
        input_dict = {}
        # low_bit_input_dict = {}
        rouge = evaluate.load('rouge')
        size = size if size is not None else len(dataset)
        for (high_bit, low_bit) in precision_switch_points:
            labels = []
            inputs = []
            # low_bit_inputs = []
            for data in tqdm(dataset[:size]):
                high_prec_steps = [0, 255 // 3, 255 // 3 * 2]
                mix_outputs = [None] * len(high_prec_steps)
                mix_input_ids = [None] * len(high_prec_steps)
                with torch.inference_mode():
                    input_ids = data['input_ids'].cuda()
                    input_len = input_ids.shape[1]
                    outputs = model.model(input_ids, precision=high_bit, past_key_values=None, use_cache=True)  
                    # low_bit_outputs = model.model(input_ids, precision=low_bit, past_key_values=None, use_cache=True)  
                    inputs.append(clone_past_key_values(outputs['past_key_values'][-1:], device='cpu')[0])
                    # low_bit_inputs.append(clone_past_key_values(low_bit_outputs['past_key_values'][-1:], device='cpu')[0])

                    print('1', inputs[-1][0].shape)
                    new_token = 0

                    while new_token < 256:
                        input_id = outputs['logits'][:, -1:].argmax(dim=-1)
                        outputs = model.model(input_id, 
                                                precision=high_bit, 
                                                use_cache=True, 
                                                past_key_values=outputs['past_key_values'])
                        input_ids = torch.cat([input_ids, input_id], dim=-1)
                        for i, step in enumerate(high_prec_steps):
                            if new_token == step:
                                mix_outputs[i] = {'logits': outputs['logits'].clone(), 
                                                'past_key_values': clone_past_key_values(outputs['past_key_values'])}
                                mix_input_ids[i] = input_ids.clone()
                            elif new_token > step:
                                input_id = mix_outputs[i]['logits'][:, -1:].argmax(dim=-1)
                                mix_outputs[i] = model.model(input_id, 
                                                            precision=low_bit, 
                                                            use_cache=True, 
                                                            past_key_values=mix_outputs[i]['past_key_values'])
                                mix_input_ids[i] = torch.cat([mix_input_ids[i], input_id], dim=-1)
                                
                        
                        new_token += 1
                        if model.tokenizer.eos_token_id in input_ids[0, :].tolist():
                            break
                        torch.cuda.empty_cache() 
                high_bit_tokens = model.tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=True)
                # print('input tokens:\n', model.tokenizer.decode(input_ids[0, :], skip_special_tokens=True))
                # print('high bit tokens:\n', high_bit_tokens)
                # print('fp16 output:\n', data['fp16_output'])
                high_bit_rouge = rouge.compute(predictions=[high_bit_tokens], references=[data['fp16_output']])['rougeL']
                # print(f'high bit rouge: {high_bit_rouge}')
                step_candidates = [len(high_prec_steps)]
                for i, mix_outputs_i in enumerate(mix_outputs):
                    if mix_outputs_i is None:
                        continue
                    mix_bit_tokens = model.tokenizer.decode(mix_input_ids[i][0, input_len:], skip_special_tokens=True)
                    # print(f'mix bit tokens {i}:\n', mix_bit_tokens)
                    mix_bit_rouge = rouge.compute(predictions=[mix_bit_tokens], references=[data['fp16_output']])['rougeL']
                    # print(f'mix bit rouge {i}: {mix_bit_rouge}')
                    if round(high_bit_rouge, 3) <= round(mix_bit_rouge, 3):
                        step_candidates.append(i)
                labels.append(min(step_candidates))
                print('labels', labels)
            label_dict[(high_bit, low_bit)] = labels
            input_dict[(high_bit, low_bit)] = inputs
            # low_bit_input_dict[(high_bit, low_bit)] = low_bit_inputs
            torch.cuda.empty_cache() 
        for (high_bit, low_bit) in precision_switch_points:
            count_dict = defaultdict(int)
            for label in label_dict[(high_bit, low_bit)]:
                count_dict[label] += 1
            print(f"Count for precision {high_bit}, {low_bit}: {count_dict}")
        self.input_dict = input_dict
        # self.low_bit_input_dict = low_bit_input_dict
        self.label_dict = label_dict
        self.precisions = precision_switch_points[0]

    def set_precision(self, precision):
        self.precisions = precision

    def __len__(self):
        return len(self.input_dict[self.precisions])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_dict[self.precisions][i],
            labels=self.label_dict[self.precisions][i],
        )



class VersionTwoSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        dataset (List[Dict[str, torch.Tensor]]): The dataset to fine-tune on.
        model (AnyPrecisionForCausalLM): The model to fine-tune.
        alpha (float): The alpha value for the key value diff threshold.
    """

    def __init__(self, dataset, model, precision_switch_points, size=None):
        super(SupervisedDataset, self).__init__()
        label_dict = {}
        input_dict = {}
        rouge = evaluate.load('rouge')
        size = size if size is not None else len(dataset)
        for (high_bit, low_bit) in precision_switch_points:
            labels = []
            inputs = []
            for data in tqdm(dataset[:size]):
                high_prec_steps = [0, 255 // 3, 255 // 3 * 2]
                mix_outputs = [None] * len(high_prec_steps)
                mix_input_ids = [None] * len(high_prec_steps)
                with torch.inference_mode():
                    input_ids = data['input_ids'].cuda()
                    input_len = input_ids.shape[1]
                    outputs = model.model(input_ids, precision=high_bit, past_key_values=None, use_cache=True)  
                    new_token = 0

                    while new_token < 256:
                        input_id = outputs['logits'][:, -1:].argmax(dim=-1)
                        outputs = model.model(input_id, 
                                                precision=high_bit, 
                                                use_cache=True, 
                                                past_key_values=outputs['past_key_values'])
                        input_ids = torch.cat([input_ids, input_id], dim=-1)
                        for i, step in enumerate(high_prec_steps):
                            if new_token == step:
                                mix_outputs[i] = {'logits': outputs['logits'].clone(), 
                                                'past_key_values': clone_past_key_values(outputs['past_key_values'])}
                                mix_input_ids[i] = input_ids.clone()
                            elif new_token > step:
                                input_id = mix_outputs[i]['logits'][:, -1:].argmax(dim=-1)
                                mix_outputs[i] = model.model(input_id, 
                                                            precision=low_bit, 
                                                            use_cache=True, 
                                                            past_key_values=mix_outputs[i]['past_key_values'])
                                mix_input_ids[i] = torch.cat([mix_input_ids[i], input_id], dim=-1)
                                
                        
                        new_token += 1
                        if model.tokenizer.eos_token_id in input_ids[0, :].tolist():
                            break
                        torch.cuda.empty_cache() 
                high_bit_tokens = model.tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=True)
                # print('input tokens:\n', model.tokenizer.decode(input_ids[0, :], skip_special_tokens=True))
                # print('high bit tokens:\n', high_bit_tokens)
                # print('fp16 output:\n', data['fp16_output'])
                high_bit_rouge = rouge.compute(predictions=[high_bit_tokens], references=[data['fp16_output']])['rougeL']
                # print(f'high bit rouge: {high_bit_rouge}')
                for i, mix_outputs_i in enumerate(mix_outputs):
                    if mix_outputs_i is None:
                        continue
                    mix_bit_tokens = model.tokenizer.decode(mix_input_ids[i][0, input_len:], skip_special_tokens=True)
                    # print(f'mix bit tokens {i}:\n', mix_bit_tokens)
                    mix_bit_rouge = rouge.compute(predictions=[mix_bit_tokens], references=[data['fp16_output']])['rougeL']
                    # print(f'mix bit rouge {i}: {mix_bit_rouge}')
                    labels.append(round(high_bit_rouge, 2) <= round(mix_bit_rouge, 2))
                    inputs.append(clone_past_key_values(mix_outputs_i['past_key_values'][-1:], device='cpu')[0])
            label_dict[(high_bit, low_bit)] = labels
            input_dict[(high_bit, low_bit)] = inputs
            torch.cuda.empty_cache() 
        for (high_bit, low_bit) in precision_switch_points:
            positive_count = sum(label_dict[(high_bit, low_bit)])
            negative_count = len(label_dict[(high_bit, low_bit)]) - positive_count
            print(f"Positive count for precision {high_bit}, {low_bit}: {positive_count}")
            print(f"Negative count for precision {high_bit}, {low_bit}: {negative_count}")
        self.input_dict = input_dict
        self.label_dict = label_dict
        self.precisions = precision_switch_points[0]

    def set_precision(self, precision):
        self.precisions = precision

    def __len__(self):
        return len(self.input_dict[self.precisions])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_dict[self.precisions][i],
            labels=self.label_dict[self.precisions][i],
        )


class VersionOneSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        dataset (List[Dict[str, torch.Tensor]]): The dataset to fine-tune on.
        model (AnyPrecisionForCausalLM): The model to fine-tune.
        alpha (float): The alpha value for the key value diff threshold.
    """

    def __init__(self, dataset, model, time_steps=1, top_k=1):
        super(VersionOneSupervisedDataset, self).__init__()

        print("Formatting inputs...")
        labels = []
        inputs = []
        for data in tqdm(dataset):
          label_dict = {}
          input_dict = {}
          for (high_bit, low_bit, _) in model.classifiers:
              with torch.inference_mode():
                high_bit_output = model(data['input_ids'].cuda()[:, :-1], precision=high_bit, use_cache=True, return_dict=True)
                high_bit_past_key_values = high_bit_output.past_key_values
                mix_bit_past_key_values = tuple(
                      tuple(past_key_value.clone() for past_key_value in layer)
                      for layer in high_bit_output.past_key_values
                    )
                input_dict[(high_bit, low_bit)] = tuple(
                      past_key_value.cpu().clone() for past_key_value in high_bit_past_key_values[-1]
                    )
                label_dict[(high_bit, low_bit)] = 1
                for _ in range(time_steps):
                    input_id = high_bit_output.logits[:, -1:].argmax(dim=-1).clone()
                    high_bit_output = model(input_id, 
                                            precision=high_bit, 
                                            use_cache=True, 
                                            return_dict=True,
                                            past_key_values=high_bit_past_key_values)
                    mix_bit_output = model(input_id, 
                                            precision=low_bit, 
                                            use_cache=True, 
                                            return_dict=True,
                                            past_key_values=mix_bit_past_key_values)
                    high_bit_token = high_bit_output.logits[:, -1].topk(top_k, dim=-1)
                    mix_bit_token = mix_bit_output.logits[:, -1].topk(top_k, dim=-1)
                    high_bit_past_key_values = high_bit_output.past_key_values
                    mix_bit_past_key_values = mix_bit_output.past_key_values
                    label_dict[(high_bit, low_bit)] &= (high_bit_token.indices == mix_bit_token.indices).all().cpu().item()
                torch.cuda.empty_cache()  
          labels.append(label_dict)
          inputs.append(input_dict)
        for (high_bit, low_bit, _) in model.classifiers:
            positive_count = sum(label[(high_bit, low_bit)] for label in labels)
            negative_count = len(labels) - positive_count
            print(f"Positive count for precision {high_bit}, {low_bit}: {positive_count}")
            print(f"Negative count for precision {high_bit}, {low_bit}: {negative_count}")
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.inputs[i],
            labels=self.labels[i],
        )


def preprocess_dataset(dataset, start_index, end_index, num_labels=4, all_layers=False):
    samples = [[] for _ in range(num_labels)]
    for i in range(start_index, end_index):
        data = {
            'input_ids': dataset[i]['input_ids'] if not all_layers else dataset[i]['input_all_layers'],
            'labels': [1 if dataset[i]['labels'] == j else 0 for j in range(num_labels)]
        }
        samples[dataset[i]['labels']].append(data)
    print(f"Number of samples for each label: {[len(samples[i]) for i in range(num_labels)]}")
    
    # balance dataset 
    majority_label = max([len(samples[i]) for i in range(num_labels)])
    balanced_dataset = []
    for i in range(num_labels):
        s = samples[i]
        if len(s) == 0:
            continue
        samples[i] = s * (majority_label // len(s)) + random.sample(s, majority_label % len(s))
        balanced_dataset += samples[i]

    random.shuffle(balanced_dataset)  
    return balanced_dataset


def train(params, config, dataset, precisions, all_layers):
    print('Training with params:', params)
    
    # KVCache Scheduler
    precision_switch_points = set()
    for p in sorted(precisions, reverse=True):
        if p-1 in precisions:
            precision_switch_points.add((p, p-1))
    scheduler = Scheduler['kv_cache'](precisions, 
                                      precision_switch_points, 
                                      dim=config.hidden_size // config.num_attention_heads,
                                      num_heads=config.num_key_value_heads,
                                      dropout_prob=params['dropout_rate']
                                      )

    ## For act scheduler
    # dim = config.hidden_size if not all_layers else dataset[0]['input_all_layers'].size(0)

    # scheduler = Scheduler['act'](precisions, 
    #                                   precision_switch_points, 
    #                                   dim=dim,
    #                                   all_layers=all_layers,
    #                                   )

    # Load data
    train_dataset_index = len(dataset) // 10

    accumulation_steps = params['gradient_accumulation_steps']
    parameters = [param for c in scheduler.classifiers.values() for param in c.parameters()]
    print(f"Number of parameters: {sum([param.numel() for param in parameters])}")
    ### NOTE: might want to use AdamW instead for correct weight decay implementation
    optimizer = optim.Adam(parameters, lr=params['learning_rate'], weight_decay=params['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss(torch.tensor([1, 0.62, 0.6, 0.7], dtype=torch.float32).cuda())

    for ((high_bit, low_bit), classifier) in scheduler.classifiers.items():
        dataset.set_precision((high_bit, low_bit))
        
        train_dataset = preprocess_dataset(dataset, train_dataset_index, len(dataset), all_layers=all_layers)
        eval_dataset = preprocess_dataset(dataset, 0, train_dataset_index, all_layers=all_layers)
        print(f"Training dataset size: {len(train_dataset)}")
        # print(f"Validation dataset size: {len(eval_dataset)}")
        
        classifier.cuda()
        for epoch in range(int(params['num_train_epochs'])):
            avg_train_loss = 0
            classifier.train()
            for i, data in enumerate(train_dataset):
                ## kvcache
                past_key_values = data['input_ids']
                labels = data['labels']
                past_key = past_key_values[0]
                past_value = past_key_values[1]
                pred = classifier(past_key.cuda(), past_value.cuda())[0]
                
                ## acts
                # acts = data['input_ids']
                # labels = data['labels']
                # pred = classifier(acts.cuda())[0]
                
                label = torch.tensor(labels, dtype=pred.dtype).cuda()
                loss = criterion(pred, label)
                avg_train_loss += loss.cpu().item()
                loss = loss / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0 or i == len(train_dataset) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
            avg_train_loss /= len(train_dataset)
            print(f"Epoch {epoch}, Training Loss {avg_train_loss}", flush=True)
        
            # Evaluate the model
            classifier.eval()
            eval_loss = 0
            with torch.no_grad():
                for data in eval_dataset:
                    ## kv cache
                    past_key_values = data['input_ids']
                    labels = data['labels']
                    past_key = past_key_values[0]
                    past_value = past_key_values[1]

                    ## acts
                    # acts = data['input_ids']
                    # labels = data['labels']
                    # pred = classifier(acts.cuda())[0]
                    pred = classifier(past_key.cuda(), past_value.cuda())[0]
                    label = torch.tensor(labels, dtype=pred.dtype).cuda()
                    eval_loss += criterion(pred, label).cpu().item()
            eval_loss /= len(eval_dataset)
            print(f"Epoch {epoch}, Validation Loss {eval_loss}", flush=True)

    return {'loss': eval_loss, 'status': STATUS_OK, 'model': scheduler.classifiers}

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_path', type=str, required=True)
    argparser.add_argument('--data_path', type=str, required=True)
    argparser.add_argument('--precisions', type=str, default='2,3,4')
    argparser.add_argument('--num_train_epochs', type=int, default=5)
    argparser.add_argument('--gradient_accumulation_steps', type=int, default=32)
    argparser.add_argument('--learning_rate', type=float, default=1e-3)
    argparser.add_argument('--weight_decay', type=float, default=0.01)
    argparser.add_argument('--dropout_rate', type=float, default=0.1)
    
    args = argparser.parse_args()
    params = vars(args)
    
    config = AutoConfig.from_pretrained(args.model_path)
    train_dataset = torch.load(args.data_path)
    precision_list = [int(p) for p in args.precisions.split(',')]
    
    ret = train(params, config, train_dataset, precision_list)
    
    # Save model
    best_model = ret['model']
    model_name = args.data_path.split("/")[-1].split("train_dataset_")[-1].split(".pt")[0]
    output_dir = f"test/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    for (high_bit, low_bit), classifier in best_model.items():
        torch.save(classifier.state_dict(), f"{output_dir}/classifier_{high_bit}_{low_bit}.pt")
    