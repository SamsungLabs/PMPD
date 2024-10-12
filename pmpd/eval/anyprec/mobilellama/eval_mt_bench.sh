python3 pmpd/eval/eval.py \
    --model-id MobileLLaMA \
    --model-path anyprec-MobileLLaMA-1.4B-Chat-4-2 \
    --fp-model mtgv/MobileLLaMA-1.4B-Chat \
    --bench-name mt_bench \
    --question-end 1000 \
    --precision-high 4 \
    --precision-low 3 \
    --steps 0,85,170 \
    --gpus 1,2,3 \
    --answer-file-dir data/anyprec/ \
    --classifier-path test/anyprec-MobileLLaMA-1.4B-Chat-4-2_4_3_256/ \
    --high-bit-steps 85 \
    --prior-distribution 0.30078125 0.33984375 0.21875 0.140625 \
    --random-prior \
    --random-uniform \
    # --kv-scheduler \
    # --baseline \
    # --static-scheduler \
    # --fp16 \
    # --search \
    # --confidence-scheduler

