#!/bin/bash

# SFT for Qwen3.5-9B on rl-rag-2/sft_ablations_bc_only_v1
# 4 nodes x 8 GPUs = 32 GPUs, SP=2, 131k seq len, 8 epochs, LINEAR schedule, LR 1e-4.
# Follow-up to the 5e-5 / 5-epoch runs (loss still decreasing at epoch 5, no instability).
# Keeps last 5 epoch checkpoints for downstream eval sweeps.

BEAKER_IMAGE="${1:-shashankg/open_instruct_auto}"

MODEL="Qwen/Qwen3.5-9B"
TOKENIZER="Qwen/Qwen3.5-9B"

DATASET="rl-rag-2/sft_ablations_bc_only_v1"

EXP_NAME="drtulu_sft_qwen35_9b_128k_8ep_linear_sp2_lr1e4"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/oe-agents \
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
    --exp_name $EXP_NAME \
    --model_name_or_path $MODEL \
    --tokenizer_name $TOKENIZER \
    --sequence_parallel_size 2 \
    --max_seq_length 131072 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --use_liger_kernel \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 8 \
    --dataset_mixer_list \
        $DATASET 1.0 \
    --dataset_mixer_list_splits \
        train \
    --gradient_checkpointing \
    --checkpointing_steps epoch \
    --keep_last_n_checkpoints 5 \
    --clean_checkpoints_at_end false \
    --timeout 7200 \
    --try_launch_beaker_eval_jobs false \
    --push_to_hub false \
    --report_to wandb \
    --with_tracking \
    --wandb_project_name oe-general-agents \
    --logging_steps 1 \
    --seed 42
