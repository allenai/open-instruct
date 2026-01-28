#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
BATCH_SIZE="${2:-4}"

uv run python mason.py \
    --cluster ai2/saturn \
    --cluster ai2/jupiter \
    --description "DPO checkpointing=full, batch_size=${BATCH_SIZE}" \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --gpus 1 -- torchrun --nproc_per_node=1 open_instruct/dpo.py \
    --model_name_or_path allenai/OLMo-2-0425-1B \
    --tokenizer_name allenai/OLMo-2-0425-1B \
    --max_seq_length 1024 \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --gradient_checkpointing_mode full \
    --learning_rate 5e-07 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --output_dir output/dpo_checkpointing_full_bs${BATCH_SIZE}/ \
    --logging_steps 1 \
    --mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 100 \
    --chat_template_name olmo \
    --seed 123 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false \
    --with_tracking
