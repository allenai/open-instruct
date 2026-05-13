#!/bin/bash

# SFT on Qwen3-4B-Instruct-2507 using hamishivi/tmax-sft-full-20260317
# 20k seq len, 2e-5 LR, 4 nodes x 8 GPUs

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

echo "Using Beaker image: $BEAKER_IMAGE"

DATASET=osieosie/tmax-sft-skill-tax-20260505-2.2k-combined-balanced-qwen3.6-27b-thinking
DATASET_CONFIG=skill_tax_20260505_2.2k_combined_balanced_thinking_all

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 4 \
    --gpus 8 \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name sft_qwen35_4b_base_our_sft \
    --model_name_or_path Qwen/Qwen3.5-4B-Base \
    --tokenizer_name Qwen/Qwen3.5-4B-Base \
    --sequence_parallel_size 4 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --dataset_mixer_list \
        $DATASET 1.0 \
    --dataset_mixer_list_config_names \
        $DATASET_CONFIG \
    --dataset_mixer_list_splits \
        train \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --seed 42
