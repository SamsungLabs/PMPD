python3 pmpd/eval/eval.py \
    --model-id phi-1_5 \
    --model-path anyprec-phi-1_5-4-2 \
    --fp-model microsoft/phi-1_5 \
    --bench-name gsm8k \
    --question-end 1000 \
    --precision-high 4 \
    --precision-low 3 \
    --steps 0,85,170 \
    --gpus 1,2,3 \
    --answer-file-dir data/anyprec/ \
    --classifier-path test/anyprec-phi-1_5-4-2_4_3_256/ \
    --high-bit-steps 85 \
    --kv-scheduler \
    --static-search \
    --fp16 \
    --baseline \
    # --static-scheduler \
    # --confidence-scheduler

