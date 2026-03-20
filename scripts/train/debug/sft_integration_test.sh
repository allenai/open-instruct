#!/bin/bash

CHECKPOINT=allenai/OLMo-2-0325-32B-DPO

LAUNCH_CMD="torchrun \
    --nproc_per_node=1 \
    open_instruct/olmo_core_finetune.py \
    --model_name_or_path $CHECKPOINT \
    --dataset_mixer_list allenai/tulu-3-sft-personas-algebra 100 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-06 \
    --warmup_ratio 0.03 \
    --num_train_epochs 2 \
    --logging_steps 1 \
    --seed 123 \
    --output_dir output/ \
    --chat_template_name tulu \
    --tokenizer_name_or_path $CHECKPOINT"

if [ -n "$1" ]; then
    BEAKER_IMAGE="$1"
    echo "Using Beaker image: $BEAKER_IMAGE"

    uv run python mason.py \
        --cluster ai2/jupiter \
        --workspace ai2/open-instruct-dev \
        --priority normal \
        --image "$BEAKER_IMAGE" \
        --description "Single GPU OLMo-core SFT test." \
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
