#!/bin/bash

# Define the GPU IDs
GPUS=(0 1 2 3)

# Define the high-bit steps
STEPS=(0 11 22)

# Function to run the job
run_job() {
  local step=$1
  local gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu python3  pmpd/eval/evaluate_generation.py --model-path gptq-MobileLlama-config.json --model-id MobileLlama --bench-name dsum --question-end 1000 --answer-file data/gptq/dsum/MobileLlama-schedule-${step}.jsonl --high-bit-steps $step --precisions 4,3 --use-multi-model --max-new-tokens 33
}

# Distribute jobs across GPUs
for ((i=0; i<${#STEPS[@]}; i++)); do
  gpu_index=$((i % ${#GPUS[@]}))
  run_job ${STEPS[$i]} ${GPUS[$gpu_index]} &
  
  # Limit the number of background jobs to the number of GPUs
  # if (( (i + 1) % ${#GPUS[@]} == 0 )); then
  #   wait
  # fi
done


# Run the first command on GPU 0
# CUDA_VISIBLE_DEVICES=0 python3  pmpd/eval/evaluate_generation.py --model-path gptq-MobileLlama-config.json --model-id MobileLlama --bench-name dsum --question-end 1000 --answer-file data/gptq/dsum/MobileLlama-2bit.jsonl --high-bit-steps 0 --precisions 2 --use-multi-model &

# Run the second command on GPU 1
CUDA_VISIBLE_DEVICES=0 python3  pmpd/eval/evaluate_generation.py --model-path gptq-MobileLlama-config.json --model-id MobileLlama --bench-name dsum --question-end 1000 --answer-file data/gptq/dsum/MobileLlama-3bit.jsonl --high-bit-steps 0 --precisions 3 --use-multi-model --max-new-tokens 33 &

# Run full precision on GPU 2
CUDA_VISIBLE_DEVICES=3 python3  pmpd/eval/evaluate_generation.py --model-path gptq-MobileLlama-config.json --model-id MobileLlama --bench-name dsum --question-end 1000 --answer-file data/gptq/dsum/MobileLlama-4bit.jsonl --high-bit-steps 0 --precisions 4 --use-multi-model --max-new-tokens 33 &

# CUDA_VISIBLE_DEVICES=2 python3  pmpd/eval/evaluate_generation.py --model-path gptq-MobileLlama-config.json --model-id MobileLlama --bench-name dsum --question-end 1000 --answer-file data/gptq/dsum/MobileLlama-2bit-3prefill.jsonl --high-bit-steps 0 --precisions 3,2 --use-multi-model &

wait

# Scheduler kv_cache
CUDA_VISIBLE_DEVICES=1 python3  pmpd/eval/evaluate_generation.py --model-path gptq-MobileLlama-config.json --model-id MobileLlama --bench-name dsum --question-end 1000 --answer-file data/gptq/dsum/MobileLlama-kv_cache-4-3.jsonl --scheduler kv_cache --classifier_path test/anyprec-MobileLLaMA-1.4B-Chat-4-2_4_3_256/ --use-multi-model --precisions 4,3 --max-new-tokens 33 &

# CUDA_VISIBLE_DEVICES=1 python3  pmpd/eval/evaluate_generation.py --model-path gptq-MobileLlama-config.json --model-id MobileLlama --bench-name dsum --question-end 100 --answer-file data/gptq/dsum/MobileLlama-kv_cache-3-2.jsonl --scheduler kv_cache --precisions 3,2 --classifier_path test_gptq-MobileLlama-config.json_lr_0.1/ &
