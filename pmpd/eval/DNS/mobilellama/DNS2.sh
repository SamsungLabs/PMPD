AHF_NAME=anyprec-MobileLLaMA-1.4B-Chat-dns-3.65-4.06-w4_orig2-gc1-c4_s100_blk512_v2
MODEL_ID=MobileLLaMA
TASK=IWSLT
PREC=3
CUDA_VISIBLE_DEVICES=1 python pmpd/eval/evaluate_generation.py --model-path DNS/${AHF_NAME} --model-id ${MODEL_ID} --bench-name ${TASK} --answer-file data/${AHF_NAME}_${TASK}.jsonl --precisions ${PREC} --max-new-tokens 60 