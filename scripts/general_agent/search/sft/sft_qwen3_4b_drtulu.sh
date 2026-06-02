#!/bin/bash

# SFT on Qwen3-4B-Instruct-2507 using rl-rag/browsecomp-gptoss-clean-qwen35-sft
# 10k seq len, 2e-5 LR, 1 nodes x 8 GPUs
# To use flash attention instead, pass --use_flash_attn

BEAKER_IMAGE="${1:-shashankg/open_instruct_auto}"
MODEL="Qwen/Qwen3-4B-Instruct-2507"
TOKENIZER="Qwen/Qwen3-4B-Instruct-2507"
DATASET=rl-rag/browsecomp-gptoss-clean-qwen35-sft
# DATASET="hamishivi/sft_ablations_bc_only_v1_sanitized"


uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/general-tool-use \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --gpus 8 \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name drtulu_sft_qwen3_4b \
    --model_name_or_path $MODEL \
    --tokenizer_name $TOKENIZER \
    --sequence_parallel_size 2 \
    --max_seq_length 10240 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --use_liger_kernel \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --add_bos \
    --dataset_mixer_list \
        $DATASET 1.0 \
    --dataset_mixer_list_splits \
        train \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --wandb_project_name oe-general-agents \
    --logging_steps 1 \
    --seed 42
