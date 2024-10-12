AHF_NAME=anyprec-vicuna-7b-v1.5-dns-2.43-2.6878-w4_orig2-gc1-c4_s100_blk512
MODEL_ID=vicuna
TASK=cnn_dm
PREC=2
CUDA_VISIBLE_DEVICES=2 python pmpd/eval/evaluate_generation.py --model-path DNS/${AHF_NAME} --model-id ${MODEL_ID} --bench-name ${TASK} --answer-file data/${AHF_NAME}_${TASK}.jsonl --precisions ${PREC} --max-new-tokens 120 &

AHF_NAME=anyprec-vicuna-7b-v1.5-dns-2.39-2.4375-w4_orig2-gc1-c4_s100_blk512
MODEL_ID=vicuna
TASK=cnn_dm
PREC=2
CUDA_VISIBLE_DEVICES=3 python pmpd/eval/evaluate_generation.py --model-path DNS/${AHF_NAME} --model-id ${MODEL_ID} --bench-name ${TASK} --answer-file data/${AHF_NAME}_${TASK}.jsonl --precisions ${PREC} --max-new-tokens 120 &

AHF_NAME=anyprec-vicuna-7b-v1.5-dns-2.74-4.625-w4_orig2-gc1-c4_s100_blk512
MODEL_ID=vicuna
TASK=dsum
PREC=2
CUDA_VISIBLE_DEVICES=0 python pmpd/eval/evaluate_generation.py --model-path DNS/${AHF_NAME} --model-id ${MODEL_ID} --bench-name ${TASK} --answer-file data/${AHF_NAME}_${TASK}.jsonl --precisions ${PREC} --max-new-tokens 33 &
