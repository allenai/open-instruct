#!/bin/bash
# SFT for Qwen3.5-7B on tmax datasets (2 x 4 nodes x 8 GPUs = 32 GPUs).
# Datasets:
#   hamishivi/tmax-sft-full-20260403  (sft_w_incomplete)
#   hamishivi/tmax-sft-full-20260317  (sft)
#
# NOTE: verify MODEL below — Qwen3.5 public sizes are 0.6B/1.7B/4B/7B/14B/32B/72B;
# update to the correct HF repo name if a 9B variant exists.

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
MODEL="Qwen/Qwen3.5-7B"

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
    --exp_name qwen35_9b_tmax_sft \
    --model_name_or_path $MODEL \
    --tokenizer_name $MODEL \
    --use_flash_attn \
    --max_seq_length 20480 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --dataset_mixer_list \
        hamishivi/tmax-sft-full-20260403 1.0 \
        hamishivi/tmax-sft-full-20260317 1.0 \
    --add_bos \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --seed 42 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
