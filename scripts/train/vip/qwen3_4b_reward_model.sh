#!/bin/bash
# Train Qwen3-4B-Base reward model on Skywork + Tulu preference data
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/reward_modeling.py \
    --exp_name qwen3_4b_base_rm \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --tokenizer_name Qwen/Qwen3-4B-Base \
    --dataset_mixer_list Skywork/Skywork-Reward-Preference-80K-v0.2 1.0 allenai/llama-3.1-tulu-3-70b-preference-mixture 1.0 \
    --max_token_length 4096 \
    --max_prompt_token_length 2048 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-6 \
    --lr_scheduler_type linear \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --with_tracking \
    --seed 1
