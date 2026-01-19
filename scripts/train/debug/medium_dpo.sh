#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
MODEL_NAME=allenai/Olmo-3-1025-7B
LR=1e-6
EXP_NAME=olmo3-7b-DPO-debug-32k-${LR}

uv run python mason.py \
    --cluster ai2/jupiter \
    --description "2 node DPO run with OLMo3-7B, 16k sequence length." \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 2 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name "$EXP_NAME" \
    --model_name_or_path "$MODEL_NAME" \
    --chat_template_name olmo \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate "$LR" \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --output_dir output/dpo_olmo3_debug/ \
    --mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 1000 \
    --seed 123 \
    --use_flash_attn \
    --logging_steps 1 \
    --loss_type dpo_norm \
    --beta 5 \
    --gradient_checkpointing \
    --with_tracking
