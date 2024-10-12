python3 pmpd/eval/eval.py \
    --model-id Qwen2-1.5B \
    --model-path anyprec-Qwen2-1.5B-4-2 \
    --fp-model Qwen/Qwen2-1.5B-Instruct \
    --bench-name mt_bench \
    --question-end 1000 \
    --precision-high 3 \
    --precision-low 2 \
    --steps 0,85,170 \
    --gpus 1,2,3 \
    --answer-file-dir data/anyprec/ \
    --classifier-path test/anyprec-Qwen2-1.5B-4-2_3_2_256/ \
    --high-bit-steps 85 \
    --static-search \
    --baseline \
    # --kv-scheduler \
    # --fp16 \
    # --static-scheduler \
    # --confidence-scheduler

