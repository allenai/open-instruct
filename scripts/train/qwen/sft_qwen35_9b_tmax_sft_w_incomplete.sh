#!/bin/bash
# SFT for Qwen3.5-9B on hamishivi/tmax-sft-full-20260403 (sft_w_incomplete).
# 4 nodes x 8 GPUs = 32 GPUs, SP=2, 32k seq len.

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
MODEL="Qwen/Qwen3.5-9B"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/tulu-3-dev \
    --priority high \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 4 \
    --budget ai2/oe-adapt \
    --gpus 8 \
    --no_auto_dataset_cache \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name qwen35_9b_tmax_sft_w_incomplete \
    --model_name_or_path $MODEL \
    --tokenizer_name $MODEL \
    --use_flash_attn \
    --use_liger_kernel \
    --max_seq_length 32768 \
    --sequence_parallel_size 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --dataset_mixer_list hamishivi/tmax-sft-full-20260403 1.0 \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --seed 42 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
