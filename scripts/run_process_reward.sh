#!/bin/bash
NUM_GPU=8
model_name=EleutherAI/llemma_34b #AI-MO/NuminaMath-7B-CoT # # #deepseek-ai/deepseek-math-7b-instruct
DESC="math_reward_model_${model_name}_on_numina_math_gsm8k_v3"

#    --image costah/open_instruct_dev --pure_docker_mode \
# python mason.py \
#     --cluster ai2/pluto-cirrascale \
#     --priority normal \
#     --budget ai2/allennlp \
#     --workspace ai2/tulu-2-improvements \
#     --description $DESC \
#     --gpus $NUM_GPU -- accelerate launch --num_machines 1 --num_processes $NUM_GPU --config_file configs/ds_configs/deepspeed_zero3.yaml \
#     open_instruct/reward_modeling.py \
#     --dataset_mixer '{"ai2-adapt-dev/numina_math_gsm8k_minerva_RM": 1.0}' \
#     --dataset_train_splits train \
#     --dataset_eval_mixer '{"ai2-adapt-dev/numina_math_gsm8k_minerva_RM": 1.0}' \
#     --dataset_eval_splits test \
#     --model_name_or_path $model_name \
#     --chat_template simple_concat_with_space \
#     --learning_rate 3e-6 \
#     --gradient_checkpointing \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --max_token_length 4096 \
#     --max_prompt_token_lenth 512 \
#     --num_train_epochs 1 \
#     --output_dir outputs/rm/rm_math_7b \
#     --with_tracking \
#     # --push_to_hub

#ai2-adapt-dev/Math-Shepherd-PRM-format
# "ai2-adapt-dev/numina_math_gsm8k_minerva_RM"

model_name=deepseek-ai/deepseek-math-7b-instruct
DESC="math__PRM_modeling_${model_name}_on_math_shepherd_8_2epoch"
NUM_GPU=8
# accelerate launch --num_processes $NUM_GPU --config_file configs/ds_configs/deepspeed_zero2.yaml open_instruct/reward_modeling_v2.py \
python mason.py \
    --cluster ai2/pluto-cirrascale \
    --priority normal \
    --budget ai2/allennlp \
    --workspace ai2/tulu-2-improvements \
    --description $DESC \
    --gpus $NUM_GPU -- accelerate launch --num_machines 1 --num_processes $NUM_GPU --config_file configs/ds_configs/deepspeed_zero3.yaml open_instruct/reward_modeling_v2.py \
    --dataset_mixer '{"ai2-adapt-dev/Math-Shepherd-PRM-chat-reformatted": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"ai2-adapt-dev/Math-Shepherd-PRM-chat-reformatted": 1.0}' \
    --dataset_eval_splits test \
    --model_name_or_path $model_name \
    --chat_template simple_concat_with_space \
    --learning_rate 3e-6 \
    --gradient_checkpointing \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --max_token_length 4096 \
    --max_prompt_token_lenth 512 \
    --num_train_epochs 2 \
    --output_dir ./outputs/rm/rm_math_7b \
    --chat_template simple_prm
    # --sanity_check \
#     --with_tracking \
