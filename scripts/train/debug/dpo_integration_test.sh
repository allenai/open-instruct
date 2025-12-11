#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

uv run python mason.py \
    --cluster ai2/neptune \
    --cluster ai2/saturn \
    --cluster ai2/jupiter \
    --cluster ai2/prior \
    --description "Single GPU DPO run, for debugging purposes." \
    --workspace ai2/tulu-thinker \
    --priority high \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --gpus 1 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    open_instruct/dpo_tune_cache.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --tokenizer_name Qwen/Qwen3-0.6B \
    --use_flash_attn false \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-07 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 3 \
    --output_dir output/dpo_pythia_14m/ \
    --logging_steps 1 \
    --dataset_mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 100 \
    --add_bos \
    --chat_template_name olmo \
    --seed 123
