# #!/bin/bash

# sleep for 9 hours

sleep 32400

CUDA_VISIBLE_DEVICES=0 python3  pmpd/eval/evaluate_generation.py --model-path gptq-llama-7b-config.json --model-id vicuna-7b --bench-name cnn_dm --question-end 1000 --answer-file data/gptq/cnn_dm/vicuna-7b-2bit-3prefill.jsonl --high-bit-steps 0 --precisions 3,2 --use-multi-model & 

CUDA_VISIBLE_DEVICES=1 python3  pmpd/eval/evaluate_generation.py --model-path gptq-llama-7b-config.json --model-id vicuna-7b --bench-name cnn_dm --question-end 1000 --answer-file data/gptq/cnn_dm/vicuna-7b-2bit-8prefill.jsonl --high-bit-steps 0 --precisions 8,2 --use-multi-model &

# MT Bench 

CUDA_VISIBLE_DEVICES=2 python3  pmpd/eval/evaluate_generation.py --model-path gptq-vicuna-7b-config.json --model-id vicuna-7b --bench-name mt_bench --answer-file data/gptq/mt_bench/vicuna-7b-2bit-3prefill.jsonl --high-bit-steps 0 --precisions 3,2 --use-multi-model &

CUDA_VISIBLE_DEVICES=3 python3  pmpd/eval/evaluate_generation.py --model-path gptq-vicuna-7b-config.json --model-id vicuna-7b --bench-name mt_bench --answer-file data/gptq/mt_bench/vicuna-7b-2bit-8prefill.jsonl --high-bit-steps 0 --precisions 8,2 --use-multi-model &