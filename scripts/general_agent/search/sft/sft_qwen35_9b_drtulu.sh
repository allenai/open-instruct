#!/bin/bash

# SFT for Qwen3.5-9B on rl-rag/browsecomp-gptoss-clean-qwen35-sft.
# 1 nodes x 8 GPUs = 8 GPUs, SP=2, 10k seq len.

BEAKER_IMAGE="${1:-shashankg/open_instruct_auto}"
MODEL="Qwen/Qwen3.5-9B"
DATASET="rl-rag/browsecomp-gptoss-clean-qwen35-sft"

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
    --exp_name qwen35_9b_drtulu_sft_sample \
    --model_name_or_path $MODEL \
    --tokenizer_name $MODEL \
    --use_liger_kernel \
    --max_seq_length 10240 \
    --sequence_parallel_size 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --seed 42 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
