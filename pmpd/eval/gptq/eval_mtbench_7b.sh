#!/bin/bash

# Define the GPU IDs
GPUS=(0 1 2 3)

# Define the high-bit steps
STEPS=(0 25 50 100 150 200)

# Function to run the job
run_job() {
  local step=$1
  local gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu python3  pmpd/eval/evaluate_generation.py --model-path gptq-vicuna-7b-config.json --model-id vicuna-7b --bench-name mt_bench --answer-file data/gptq/mt_bench/4-2/vicuna-7b-schedule-${step}.jsonl --high-bit-steps $step --precisions 4,2 --use-multi-model
}

# Distribute jobs across GPUs
for ((i=0; i<${#STEPS[@]}; i++)); do
  gpu_index=$((i % ${#GPUS[@]}))
  run_job ${STEPS[$i]} ${GPUS[$gpu_index]} &
  
  # Limit the number of background jobs to the number of GPUs
  if (( (i + 1) % ${#GPUS[@]} == 0 )); then
    wait
  fi
done

# Wait for any remaining background jobs to finish
wait

# # Run the first command on GPU 1
# CUDA_VISIBLE_DEVICES=1 python3  pmpd/eval/evaluate_generation.py --model-path gptq-vicuna-7b-config.json --model-id vicuna-7b --bench-name mt_bench --answer-file data/gptq/mt_bench/vicuna-7b-2bit.jsonl --high-bit-steps 0 --precisions 2 --use-multi-model &

# # Run the second command on GPU 2
# CUDA_VISIBLE_DEVICES=2 python3  pmpd/eval/evaluate_generation.py --model-path gptq-vicuna-7b-config.json --model-id vicuna-7b --bench-name mt_bench --answer-file data/gptq/mt_bench/vicuna-7b-3bit.jsonl --high-bit-steps 0 --precisions 3 --use-multi-model &

CUDA_VISIBLE_DEVICES=0 python3  pmpd/eval/evaluate_generation.py --model-path gptq-vicuna-7b-config.json --model-id vicuna-7b --bench-name mt_bench --answer-file data/gptq/mt_bench/vicuna-7b-4bit.jsonl --high-bit-steps 0 --precisions 4 --use-multi-model &

# # Run full precision on GPU 3
# CUDA_VISIBLE_DEVICES=3 python3  pmpd/eval/evaluate_generation.py --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b --bench-name mt_bench --answer-file data/gptq/mt_bench/vicuna-7b-fp16.jsonl --use-fp &

wait

# Scheduler kv_cache
CUDA_VISIBLE_DEVICES=3 python3  pmpd/eval/evaluate_generation.py --model-path gptq-vicuna-7b-config.json --model-id vicuna-7b --bench-name mt_bench --answer-file data/gptq/mt_bench/vicuna-7b-kv_cache-4-3-2.jsonl --scheduler kv_cache --classifier_path test_gptq-vicuna-7b-config.json_1024_1.1_lr_0.005/ --precisions 4,3,2 --use-multi-model &

# CUDA_VISIBLE_DEVICES=3 python3  pmpd/eval/evaluate_generation.py --model-path gptq-vicuna-7b-config.json --model-id vicuna-7b --bench-name mt_bench --answer-file data/gptq/mt_bench/vicuna-7b-kv_cache-3-2.jsonl --scheduler kv_cache --precisions 3,2 --classifier_path test_gptq-vicuna-7b-config.json_lr_0.1/ &