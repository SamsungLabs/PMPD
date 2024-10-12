python3 pmpd/eval/eval.py \
    --model-id vicuna-7b \
    --model-path anyprec-vicuna-7b-4-2 \
    --fp-model lmsys/vicuna-7b-v1.5 \
    --bench-name IWSLT \
    --question-end 1000 \
    --precision-high 3 \
    --precision-low 2 \
    --steps 0,20,40 \
    --gpus 1,2,3 \
    --answer-file-dir data/anyprec/ \
    --classifier-path test/anyprec-vicuna-7b-4-2_3_2_256/ \
    --high-bit-steps 20 \
    --max-new-tokens 60 \
    --static-search \
    # --kv-scheduler \
    # --baseline \
    # --search \
    # --fp16 \
    # --static-scheduler \
    # --confidence-scheduler

