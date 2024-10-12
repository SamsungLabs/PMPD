AHF_NAME=anyprec-MobileLLaMA-1.4B-Chat-dns-3.37-2.3125-w4_orig2-gc1-c4_s100_blk512
MODEL_ID=MobileLLaMA
TASK=cnn_dm
PREC=3
CUDA_VISIBLE_DEVICES=0 python pmpd/eval/evaluate_generation.py --model-path DNS/${AHF_NAME} --model-id ${MODEL_ID} --bench-name ${TASK} --answer-file data/${AHF_NAME}_${TASK}.jsonl --precisions ${PREC} &

AHF_NAME=anyprec-MobileLLaMA-1.4B-Chat-dns-3.19-1.1879-w4_orig2-gc1-c4_s100_blk512
MODEL_ID=MobileLLaMA
TASK=cnn_dm
PREC=3
CUDA_VISIBLE_DEVICES=0 python pmpd/eval/evaluate_generation.py --model-path DNS/${AHF_NAME} --model-id ${MODEL_ID} --bench-name ${TASK} --answer-file data/${AHF_NAME}_${TASK}.jsonl --precisions ${PREC} &

AHF_NAME=anyprec-MobileLLaMA-1.4B-Chat-dns-3.21-1.3125-w4_orig2-gc1-c4_s100_blk512
MODEL_ID=MobileLLaMA
TASK=dsum
PREC=3
CUDA_VISIBLE_DEVICES=0 python pmpd/eval/evaluate_generation.py --model-path DNS/${AHF_NAME} --model-id ${MODEL_ID} --bench-name ${TASK} --answer-file data/${AHF_NAME}_${TASK}.jsonl --precisions ${PREC} --max-new-tokens 33 &
