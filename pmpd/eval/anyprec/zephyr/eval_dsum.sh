python3 pmpd/eval/eval.py \
    --model-id zephyr \
    --model-path anyprec-stablelm-zephyr-3b \
    --fp-model stabilityai/stablelm-zephyr-3b \
    --bench-name dsum \
    --question-end 1000 \
    --precision-high 4 \
    --precision-low 3 \
    --steps 0,11,22 \
    --gpus 0,1,2 \
    --answer-file-dir data/anyprec/ \
    --classifier-path test/anyprec-stablelm-zephyr-3b_4_3_256/ \
    --high-bit-steps 20 \
    --max-new-tokens 33 \
     --prior-distribution 0.37109375 0.2421875 0.1171875 0.26953125 \
    --baseline \
    --kv-scheduler \
    --search \
    # --static-scheduler \
    # --static-search \
    # --fp16 \
    # --confidence-scheduler

