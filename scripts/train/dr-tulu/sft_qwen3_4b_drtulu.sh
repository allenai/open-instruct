#!/bin/bash

# SFT on Qwen3-4B-Instruct-2507 using rl-rag/browsecomp-gptoss-clean-qwen35-sft
# 10k seq len, 2e-5 LR, 1 nodes x 8 GPUs
# To use flash attention instead, pass --use_flash_attn

BEAKER_IMAGE="${1:-shashankg/open_instruct_auto}"
DATASET=rl-rag/browsecomp-gptoss-clean-qwen35-sft
MODEL="Qwen/Qwen3-4B-Instruct-2507"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/general-tool-use \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --env BEAKER_ALLOW_SUBCONTAINERS=1 \
    --env BEAKER_SKIP_DOCKER_SOCKET=1 \
    --budget ai2/oe-omai \
    --gpus 8 \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name sft_qwen3_4b_drtulu_sample \
    --model_name_or_path $MODEL \
    --tokenizer_name $MODEL \
    --use_liger_kernel \
    --max_seq_length 10240 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --add_bos \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --seed 42
