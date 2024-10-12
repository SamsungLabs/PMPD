python3 pmpd/eval/eval.py \
    --model-id zephyr \
    --model-path anyprec-stablelm-zephyr-3b \
    --fp-model stabilityai/stablelm-zephyr-3b \
    --bench-name mt_bench \
    --question-end 1000 \
    --precision-high 4 \
    --precision-low 3 \
    --steps 0,60,120 \
    --gpus 1,2,3 \
    --answer-file-dir data/test/ \
    --classifier-path test/anyprec-stablelm-zephyr-3b_4_3_256/ \
    --high-bit-steps 60 \
    --max-new-tokens 180 \
    --prior-distribution 0.37109375 0.2421875 0.1171875 0.26953125 \
    --kv-scheduler \
    # --random-prior \
    # --random-uniform \
    # --static-scheduler \
    # --static-search \
    # --fp16 \
    # --baseline \
    # --confidence-scheduler

