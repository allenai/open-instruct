#!/bin/bash
# Two-node OLMo-core SFT integration test.

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
CHECKPOINT=allenai/OLMo-2-0325-32B-DPO

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Two-node OLMo-core SFT test." \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 2 \
    --budget ai2/oe-adapt \
    --gpus 8 \
    --non_resumable \
    -- \
    torchrun \
    --nproc_per_node=8 \
    open_instruct/olmo_core_finetune.py \
    --model_name_or_path $CHECKPOINT \
    --dataset_mixer_list allenai/tulu-3-sft-personas-algebra 100 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-06 \
    --warmup_ratio 0.03 \
    --num_train_epochs 2 \
    --logging_steps 1 \
    --seed 123 \
    --output_dir output/ \
    --with_tracking \
    --wandb_project open_instruct_internal \
    --chat_template_name tulu \
    --tokenizer_name_or_path $CHECKPOINT
