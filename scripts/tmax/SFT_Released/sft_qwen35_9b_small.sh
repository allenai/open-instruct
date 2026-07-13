#!/bin/bash
# SFT On Qwen3.5 9B on *just* tmax data ('small' SFT)
# We use a version of Qwen 3.5 with an interleaved reasoning chat template

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

echo "Using Beaker image: $BEAKER_IMAGE"

DATASET=allenai/tmax-sft

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 4 \
    --budget ai2/oe-adapt \
    --gpus 8 \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name sft_qwen35_9b_small \
    --model_name_or_path hamishivi/Qwen3.5-9B \
    --tokenizer_name hamishivi/Qwen3.5-9B \
    --use_flash_attn \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --dataset_mixer_list $DATASET 1.0 \
    --dataset_mixer_list_config_names \
        skill_tax_20260505_2.2k_combined_balanced_thinking_all \
    --add_bos \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --seed 42
