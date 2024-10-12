from torch import nn
import torch
from any_precision import AnyPrecisionForCausalLM
from .zoo import MultiPrecModelWrapper
from auto_gptq import AutoGPTQForCausalLM
from transformers.generation.utils import ModelOutput
from transformers import AutoTokenizer
from collections import defaultdict
from pmpd.modules.scheduler import ActScheduler
from any_precision.modules.AnyPrecisionLinear import AnyPrecisionLinear
import json

def set_hook(layer_idx, acts):
    if layer_idx not in acts:
        acts[layer_idx] = {} 
    def hook(m, inp, oup):
        inp = inp[0]

        acts[layer_idx]['mean'] = inp.mean().cpu()
        acts[layer_idx]['var'] = inp.var(correction=1).cpu()
        acts[layer_idx]['acts'] = inp.detach().cpu()

    return hook

class PMPDForCausalLM(nn.Module):
  '''
  A model based on AnyPrecision that can be used for adaptive precision scheduling during inference.
  '''
  
  def __init__(self, model_path, scheduler=None, precisions=None, use_anyprec=True, quantize_model_cls=AutoGPTQForCausalLM):
    super(PMPDForCausalLM, self).__init__()
    if use_anyprec:
      self.model = AnyPrecisionForCausalLM.from_quantized(
        model_path,
        precisions=precisions
      )
      tokenizer_path = model_path
    else:
      # model path is a json file containing the paths to the quantized models
      self.model = MultiPrecModelWrapper.from_quantized(
        model_path, 
        precisions,
        quantize_model_cls=quantize_model_cls)
      tokenizer_path = self.model.config['model_path']
        
    self.model.eval().cuda()
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    self.scheduler = scheduler
  
  
  def forward(self, *args, **kwargs):
    return self.model.forward(*args, **kwargs)

  
  @torch.inference_mode()
  def generate(self, *args, **kwargs):
    assert self.scheduler is not None, "Scheduler is not provided."
    input_ids = kwargs['input_ids']
    input_ids = input_ids.clone()
    input_len = input_ids.shape[1]
    max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else (kwargs['max_length'] - input_len if 'max_length' in kwargs else 256)
    prefill_bit = kwargs['prefill_bit'] if 'prefill_bit' in kwargs else None
    past_key_values = kwargs['past_key_values'] if 'past_key_values' in kwargs else None
    
    if prefill_bit is not None:
      print(f"Prefill with bit {prefill_bit} model")
      outputs = self.model(input_ids, precision=prefill_bit, past_key_values=past_key_values, use_cache=True)  
    else:
      max_precision = max(self.model.precisions)
      # print(f"Prefill with the highest bit {max_precision} model")
      outputs = self.model(input_ids, precision=max_precision, past_key_values=past_key_values, use_cache=True)
    new_token = 0
    current_bit = max(self.scheduler.precisions)
    schedule_dict = {}
    precision_log = defaultdict(int)
    self.scheduler.reset()
    # Generation loop
    if isinstance(self.scheduler, ActScheduler):
      hooks = {}
    while True:
        input_id = outputs.logits[:, -1:].argmax(dim=-1)

        if isinstance(self.scheduler, ActScheduler):
          acts = {}
          linear_layer_idx = 0
          for m in self.model.modules():
              if isinstance(m, (nn.Linear, AnyPrecisionLinear)):
                  hooks[linear_layer_idx] = m.register_forward_hook(set_hook(linear_layer_idx, acts))
                  linear_layer_idx += 1
        outputs = self.model(input_id, precision=current_bit, use_cache=True, past_key_values=outputs.past_key_values)
        if isinstance(self.scheduler, ActScheduler):
          for h in hooks.values():
              h.remove()
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        new_token += 1

        # End generation if the EOS token is generated for any sequence in the batch
        if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > max_new_tokens:
            break
        
        # Schedule the next precision
        schedule_dict['index'] = new_token
        schedule_dict['past_key_values'] = outputs.past_key_values[-1]
        schedule_dict['logits'] = outputs.logits
        schedule_dict['precision'] = current_bit
        if isinstance(self.scheduler, ActScheduler):
          if self.scheduler.all_layers:
            _temp = []
            for layer_acts in acts.values():
              _temp.append(layer_acts['mean'])
              _temp.append(layer_acts['var'])
            schedule_dict['acts'] = torch.stack(_temp)
          else:            
            schedule_dict['acts'] = acts[list(acts.keys())[-1]]['acts']
        current_bit = self.scheduler.schedule(**schedule_dict)
        precision_log[current_bit] += 1
           
    self.scheduler.reset()
    return ModelOutput(
        input_ids=input_ids,
        new_token=new_token,
        precision_log=precision_log,
        past_key_values=outputs.past_key_values
    )
    
  
  # for gradio demo
  @torch.inference_mode()
  def pmpd_generate(self, input_ids, max_new_tokens=256):
    assert self.scheduler is not None, "Scheduler is not provided."
    input_ids = input_ids.clone()
    input_len = input_ids.shape[1]
    prefill_bit = None
    past_key_values = None
    
    if prefill_bit is not None:
      print(f"Prefill with bit {prefill_bit} model")
      outputs = self.model(input_ids, precision=prefill_bit, past_key_values=past_key_values, use_cache=True)  
    else:
      max_precision = max(self.model.precisions)
      print(f"Prefill with the highest bit {max_precision} model")
      outputs = self.model(input_ids, precision=max_precision, past_key_values=past_key_values, use_cache=True)
    new_token = 0
    current_bit = max(self.scheduler.precisions)
    schedule_dict = {}
    precision_log = defaultdict(int)
    self.scheduler.reset()
    # Generation loop
    while True:
        input_id = outputs.logits[:, -1:].argmax(dim=-1)
        outputs = self.model(input_id, precision=current_bit, use_cache=True, past_key_values=outputs.past_key_values)
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        new_token += 1
        
        yield input_ids, current_bit

        # End generation if the EOS token is generated for any sequence in the batch
        if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > max_new_tokens:
            break
        
        # Schedule the next precision
        schedule_dict['index'] = new_token
        schedule_dict['past_key_values'] = outputs.past_key_values[-1]
        schedule_dict['logits'] = outputs.logits
        schedule_dict['precision'] = current_bit
        current_bit = self.scheduler.schedule(**schedule_dict)
        precision_log[current_bit] += 1
        
    self.scheduler.reset()
    
  # for gradio demo
  @torch.inference_mode()
  def naive_generate(self, input_ids, precision, max_new_tokens=256):
    print(f"Using precision {precision}")
    input_ids = input_ids.clone()
    input_len = input_ids.shape[1]
    past_key_values = None
    outputs = self.model(input_ids, precision=precision, past_key_values=past_key_values, use_cache=True)
    new_token = 0
    # Generation loop
    while True:
        input_id = outputs.logits[:, -1:].argmax(dim=-1)
        outputs = self.model(input_id, precision=precision, use_cache=True, past_key_values=outputs.past_key_values)
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        new_token += 1
        
        yield input_ids

        # End generation if the EOS token is generated for any sequence in the batch
        if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > max_new_tokens:
            break
