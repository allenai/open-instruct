#!/bin/bash

# SFT for Qwen3-8B on rl-rag-2/sft_ablations_bc_only_v1
# 4 nodes x 8 GPUs = 32 GPUs, SP=2, 131k seq len, 5 epochs, LINEAR schedule, LR 5e-5.
# Qwen3-8B native context is only 40960, so YaRN rope_scaling (factor 4.0, original 32768
# -> 131072) is enabled via --additional_model_arguments to reach 131k with zero truncation,
# matching the Qwen3.5-9B 5e-5/5-epoch run for comparison.

BEAKER_IMAGE="${1:-shashankg/open_instruct_auto}"

# shatu/Qwen3-8B-Reasoning-Fix = Qwen3-8B with the prefix-stable chat template
# (empty-<think> loop.last injection removed; reasoning kept per-turn). Needed because the
# stock Qwen3-8B template is not prefix-stable for the SFT label-spanner (crashes caching).
MODEL="shatu/Qwen3-8B-Reasoning-Fix"
TOKENIZER="shatu/Qwen3-8B-Reasoning-Fix"

DATASET="rl-rag-2/sft_ablations_bc_only_v1"

EXP_NAME="drtulu_sft_qwen3_8b_128k_5ep_linear_sp2_yarn"

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
    --additional_model_arguments '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768,"rope_theta":1000000}}' \
    --sequence_parallel_size 2 \
    --max_seq_length 131072 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --use_liger_kernel \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 5 \
    --dataset_mixer_list \
        $DATASET 1.0 \
    --dataset_mixer_list_splits \
        train \
    --gradient_checkpointing \
    --checkpointing_steps epoch \
    --clean_checkpoints_at_end false \
    --timeout 7200 \
    --try_launch_beaker_eval_jobs false \
    --push_to_hub false \
    --report_to wandb \
    --with_tracking \
    --wandb_project_name oe-general-agents \
    --logging_steps 1 \
    --seed 42
