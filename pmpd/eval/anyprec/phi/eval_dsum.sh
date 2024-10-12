python3 pmpd/eval/eval.py \
    --model-id phi-1_5 \
    --model-path anyprec-phi-1_5-4-2 \
    --fp-model microsoft/phi-1_5 \
    --bench-name dsum \
    --question-end 1000 \
    --precision-high 4 \
    --precision-low 3 \
    --steps 0,11,22 \
    --gpus 0,1,2 \
    --answer-file-dir data/anyprec/ \
    --classifier-path test/anyprec-phi-1_5-4-2_4_3_256/ \
    --high-bit-steps 20 \
    --max-new-tokens 33 \
    --static-search \
    # --kv-scheduler \
    # --baseline \
    # --search \
    # --confidence-scheduler

