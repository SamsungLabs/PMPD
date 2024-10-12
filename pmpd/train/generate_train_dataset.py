import argparse
import random
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from any_precision import AnyPrecisionForCausalLM
from train_scheduler import SupervisedDataset
from pmpd import Scheduler, MultiPrecModelWrapper, PMPDForCausalLM
from importlib import import_module

def get_func_from_args(module: str):
    module_name, class_name = module.rsplit(".", 1)
    return getattr(import_module(module_name), class_name)

def first_stage_dataset(args):
  random.seed(42)
  c4 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
  c4 = list(c4.take(args.size))
  tokenizer = AutoTokenizer.from_pretrained(args.model_path)
  model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).eval().cuda()
  max_length = 2048
  dataset = []
  for d in tqdm(c4):
    text = tokenizer(d['text'], return_tensors='pt', truncation=True, max_length=max_length)
    range_min = min(32, text['input_ids'].shape[1] // 2)
    text['input_ids'] = text['input_ids'][:, :random.randint(range_min, text['input_ids'].shape[1])]
    input_len = text['input_ids'].shape[1]
    with torch.inference_mode():
      output = model.generate(text['input_ids'].cuda(),
                              max_new_tokens=256)
    # print(tokenizer.decode(output[0, input_len:], skip_special_tokens=True))
    dataset.append({
      'input_ids': text['input_ids'],
      'fp16_output': tokenizer.decode(output[0, input_len:], skip_special_tokens=True),
      })
    torch.cuda.empty_cache()
  dataset_name = f'1st_stage_train_dataset_{args.model_path.split("/")[-1]}_{args.size}.pt'
  torch.save(dataset, f'{args.dir}/{dataset_name}')


def second_stage_dataset(args):
    dataset = torch.load(args.dataset)
    precisions_list = [int(p) for p in args.precisions.split(',')]
    model = PMPDForCausalLM(args.model_path, 
                                precisions=precisions_list, 
                                use_anyprec=args.use_anyprec).eval().cuda()

    # Scheduler
    precision_switch_points = set()
    for p in sorted(precisions_list, reverse=True):
      if p-1 in precisions_list:
        precision_switch_points.add((p, p-1))
    precision_switch_points = list(precision_switch_points)
      
    dataset_func = get_func_from_args(args.dataset_module)
    
    supervised_dataset = dataset_func(dataset, model, precision_switch_points, size=args.size)
    prec_str = '_'.join([str(p) for p in precisions_list])
    torch.save(supervised_dataset, f'{args.dir}/2nd_stage_train_dataset_{args.model_path.split("/")[-1]}_{prec_str}_{args.size}{args.postfix_name}.pt')


def main(args):
  if not args.second_stage:
    first_stage_dataset(args)
  else:
    second_stage_dataset(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate train dataset')
    parser.add_argument('--second-stage', action='store_true', help='Generate second stage dataset', default=False)
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model')
    parser.add_argument('--size', type=int, default=256, help='Size of the dataset')
    parser.add_argument('--dir', type=str, default='.', help='Directory to save the dataset')
    parser.add_argument('--dataset', type=str, default=None, help='Path to the dataset')
    parser.add_argument('--precisions', type=str, default='2,3,4', help='Precisions to use')
    parser.add_argument('--time-steps', type=int, default=1, help='Number of time steps')
    parser.add_argument('--top-k', type=int, default=1, help='Top k tokens to consider')
    parser.add_argument('--use-anyprec', action='store_true', help='Use AnyPrecision', default=False)
    parser.add_argument('--dataset-module', type=str, help='path to dataset module', default='train_scheduler.SupervisedDataset')
    parser.add_argument('--postfix-name', type=str, help='2nd stage postfix string to avoid overwritting existing datasets', default='')
    args = parser.parse_args()

    main(args)