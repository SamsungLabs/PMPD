python3 pmpd/eval/eval.py \
    --model-id zephyr \
    --model-path anyprec-stablelm-zephyr-3b \
    --fp-model stabilityai/stablelm-zephyr-3b \
    --bench-name cnn_dm \
    --question-end 1000 \
    --precision-high 4 \
    --precision-low 3 \
    --steps 0,40,80 \
    --gpus 0,1,3 \
    --answer-file-dir data/anyprec/ \
    --classifier-path test/anyprec-stablelm-zephyr-3b_4_3_256/ \
    --high-bit-steps 80 \
    --max-new-tokens 120 \
    --prior-distribution 0.37109375 0.2421875 0.1171875 0.26953125 \
    --kv-scheduler \
    --baseline \
    --search \
    # --random-prior \
    # --random-uniform \
    # --static-scheduler \
    # --static-search \
    # --fp16 \
    # --confidence-scheduler

