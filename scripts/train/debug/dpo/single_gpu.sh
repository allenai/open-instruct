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
    --no_auto_dataset_cache \
    --gpus 1 -- torchrun --nproc_per_node=1 open_instruct/dpo.py \
    --model_name_or_path /weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-7b-instruct-sft-1115 \
    --config_name olmo3_7B \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-07 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 3 \
    --output_dir output/dpo_olmo_core_debug/ \
    --logging_steps 1 \
    --mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 100 \
    --chat_template_name olmo123 \
    --seed 123 \
    --loss_type dpo_norm \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false \
    --with_tracking \
    --gradient_checkpointing \
    --gradient_checkpointing_mode budget \
    --activation_memory_budget 0.1 \
    --compile_model
