#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

uv run python mason.py \
    --cluster ai2/saturn \
    --cluster ai2/jupiter \
    --description "Single GPU DPO run with dpo_tune_cache.py, for comparing metrics with dpo.py." \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --gpus 1 -- accelerate launch --num_processes 1 open_instruct/dpo_tune_cache.py \
    --model_name_or_path /weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-7b-instruct-sft-1115 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-07 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 3 \
    --output_dir output/dpo_tune_cache_debug/ \
    --logging_steps 1 \
    --mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 100 \
    --chat_template_name olmo123 \
    --seed 123 \
    --loss_type dpo_norm \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false \
    --with_tracking \
    --gradient_checkpointing
