#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

uv run python mason.py \
    --cluster ai2/saturn \
    --cluster ai2/jupiter \
    --description "Single GPU DPO run with OLMo-core, for debugging purposes." \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --gpus 1 -- torchrun --nproc_per_node=1 open_instruct/dpo.py \
    --model_name_or_path allenai/Olmo-3-1025-7B \
    --chat_template_name olmo \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-07 \
    --warmup_ratio 0.1 \
    --num_epochs 3 \
    --output_dir output/dpo_olmo_debug/ \
    --dataset_mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 100 \
    --seed 123 \
    --with_tracking
