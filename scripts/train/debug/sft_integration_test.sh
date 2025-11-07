#!/bin/bash

LAUNCH_CMD="accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    open_instruct/finetune.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --tokenizer_name Qwen/Qwen3-0.6B \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-06 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --logging_steps 1 \
    --model_revision main \
    --dataset_mixer_list allenai/tulu-3-sft-personas-algebra 100 \
    --add_bos \
    --seed 123 \
    --chat_template_name tulu \
    --push_to_hub false"

if [ -n "$1" ]; then
    BEAKER_IMAGE="$1"
    echo "Using Beaker image: $BEAKER_IMAGE"

    uv run python mason.py \
        --cluster ai2/jupiter \
        --workspace ai2/open-instruct-dev \
        --priority normal \
        --image "$BEAKER_IMAGE" \
        --description "Single GPU finetune job." \
        --pure_docker_mode \
        --preemptible \
        --num_nodes 1 \
        --budget ai2/oe-adapt \
        --gpus 1 \
        --non_resumable \
        -- \
        $LAUNCH_CMD
else
    echo "Running locally..."
    uv run $LAUNCH_CMD
fi
