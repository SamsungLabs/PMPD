python3 pmpd/eval/eval.py \
    --model-id vicuna-7b \
    --model-path anyprec-vicuna-7b-4-2 \
    --fp-model lmsys/vicuna-7b-v1.5 \
    --bench-name dsum \
    --question-end 1000 \
    --precision-high 3 \
    --precision-low 2 \
    --steps 0,11,22 \
    --gpus 0,1,3 \
    --answer-file-dir data/anyprec/ \
    --classifier-path test/anyprec-vicuna-7b-4-2_3_2_256/ \
    --high-bit-steps 20 \
    --max-new-tokens 33 \
    --static-search \
    # --baseline \
    # --kv-scheduler \
    # --search \
    # --static-scheduler \
    # --fp16 \
    # --confidence-scheduler

