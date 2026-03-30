#!/bin/bash

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Single GPU OLMo-core SFT job." \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --gpus 1 \
    --non_resumable \
    --no-host-networking \
    --no_auto_dataset_cache \
    -- \
    torchrun --nproc_per_node=1 \
    open_instruct/olmo_core_finetune.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-06 \
    --warmup_ratio 0.03 \
    --num_epochs 2 \
    --logging_steps 1 \
    --mixer_list allenai/tulu-3-sft-personas-algebra 100 \
    --seed 123 \
    --compile_model true \
    --with_tracking
