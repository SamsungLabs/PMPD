import argparse
import torch
import random
import time
from tqdm import tqdm
from any_precision import AnyPrecisionForCausalLM

def main():
  parser = argparse.ArgumentParser(description='Adaquant evaluation script')
  parser.add_argument('--model-path', type=str, help='Path to the AnyPrecision model')
  parser.add_argument('--precision', type=int, help='Precision to evaluate the model at')
  parser.add_argument('--steps', type=int, help='Number of steps to run the model for', default=1000)
  
  args = parser.parse_args()
  
  model = AnyPrecisionForCausalLM.from_quantized(args.model_path, precisions=[args.precision])
  model.eval()
  model.cuda()
  
  random.seed(42)
  torch.manual_seed(42)
  torch.cuda.manual_seed(42)
  
  print("Warming up the model")
  
  # warmup
  for _ in tqdm(range(100)):
    input_ids = torch.randint(0, 1024, (1, 1000)).cuda()
    with torch.no_grad():
      _ = model(input_ids, precision=args.precision)
  
  print("Warmup done")
  
  times = []
  for i in tqdm(range(args.steps)):
    input_ids = torch.randint(0, 1024, (1, 1000)).cuda()
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
      _ = model(input_ids, precision=args.precision)
    torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)
    
  print(f"Average time for {args.steps} steps: {sum(times) / len(times)}")
  
if __name__ == '__main__':
  main()
  
  
  