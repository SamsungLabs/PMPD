# CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" python pmpd/train/train_scheduler.py --model_name_or_path lmsys/vicuna-7b-v1.5 \
#     --data_path pmpd/train/data/2nd_stage_train_dataset_anyprec-vicuna-7b-4-2_2048_1.1.pt \
#     --output_dir test \
#     --num_train_epochs 30 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 0.0045 \
#     --weight_decay 0.08031288528295362 \
#     --precisions 3,2

# CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" python pmpd/train/train_scheduler.py --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
#     --data_path pmpd/train/data/2nd_stage_train_dataset_anyprec-Llama-2-7b-chat-hf-4-2_2048_1.1.pt \
#     --output_dir test \
#     --num_train_epochs 30 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 0.0045 \
#     --weight_decay 0.08031288528295362 \
#     --precisions 3,2

# CUDA_VISIBLE_DEVICES=2 NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" python pmpd/train/train_scheduler.py --model_name_or_path lmsys/vicuna-7b-v1.5 \
#     --data_path pmpd/train/data/2nd_stage_train_dataset_gptq-vicuna-7b-config.json_2048_1.1.pt \
#     --output_dir test \
#     --num_train_epochs 30 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 0.0045 \
#     --weight_decay 0.08031288528295362 \
#     --precisions 4,3,2

# CUDA_VISIBLE_DEVICES=3 NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" python pmpd/train/train_scheduler.py --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
#     --data_path pmpd/train/data/2nd_stage_train_dataset_gptq-llama-7b-config.json_2048_1.1.pt \
#     --output_dir test \
#     --num_train_epochs 30 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 0.0045 \
#     --weight_decay 0.08031288528295362 \
#     --precisions 4,3,2

CUDA_VISIBLE_DEVICES=0 python pmpd/train/hypertune.py --model_path mtgv/MobileLLaMA-1.4B-Chat --data_path pmpd/train/data/2nd_stage_train_dataset_anyprec-MobileLLaMA-1.4B-Chat-4-2_4_3_256.pt --precisions 4,3 &

CUDA_VISIBLE_DEVICES=1 python pmpd/train/hypertune.py --model_path microsoft/phi-1_5 --data_path pmpd/train/data/2nd_stage_train_dataset_anyprec-phi-1_5-4-2_4_3_256.pt --precisions 4,3 &

# CUDA_VISIBLE_DEVICES=0 python pmpd/train/hypertune.py --model_path Qwen/Qwen2-1.5B-Instruct --data_path pmpd/train/data/2nd_stage_train_dataset_anyprec-Qwen2-1.5B-4-2_4_3_256.pt --precisions 4,3 &

CUDA_VISIBLE_DEVICES=2 python pmpd/train/hypertune.py --model_path stabilityai/stablelm-zephyr-3b --data_path pmpd/train/data/2nd_stage_train_dataset_anyprec-stablelm-zephyr-3b_4_3_256.pt --precisions 4,3 &

CUDA_VISIBLE_DEVICES=3 python pmpd/train/hypertune.py --model_path lmsys/vicuna-7b-v1.5 --data_path pmpd/train/data/2nd_stage_train_dataset_anyprec-vicuna-7b-4-2_3_2_256.pt --precisions 3,2 &



