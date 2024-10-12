AHF_NAME=anyprec-phi-1_5-dns-3.71-4.4375-w4_orig2-gc1-c4_s100_blk512
MODEL_ID=phi_1-5
TASK=cnn_dm
PREC=3
CUDA_VISIBLE_DEVICES=1 python pmpd/eval/evaluate_generation.py --model-path DNS/${AHF_NAME} --model-id ${MODEL_ID} --bench-name ${TASK} --answer-file data/${AHF_NAME}_${TASK}.jsonl --precisions ${PREC} &

AHF_NAME=anyprec-phi-1_5-dns-3.30-1.875-w4_orig2-gc1-c4_s100_blk512
MODEL_ID=phi_1-5
TASK=dsum
PREC=3
CUDA_VISIBLE_DEVICES=1 python pmpd/eval/evaluate_generation.py --model-path DNS/${AHF_NAME} --model-id ${MODEL_ID} --bench-name ${TASK} --answer-file data/${AHF_NAME}_${TASK}.jsonl --precisions ${PREC} --max-new-tokens 33 &

AHF_NAME=anyprec-phi-1_5-dns-3.52-3.25-w4_orig2-gc1-c4_s100_blk512
MODEL_ID=phi_1-5
TASK=dsum
PREC=3
CUDA_VISIBLE_DEVICES=1 python pmpd/eval/evaluate_generation.py --model-path DNS/${AHF_NAME} --model-id ${MODEL_ID} --bench-name ${TASK} --answer-file data/${AHF_NAME}_${TASK}.jsonl --precisions ${PREC} --max-new-tokens 33 &