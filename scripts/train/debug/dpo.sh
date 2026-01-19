#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

uv run python mason.py \
    --cluster ai2/saturn \
    --cluster ai2/jupiter \
    --description "Single GPU DPO run, for debugging purposes." \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --gpus 1 -- torchrun --nproc_per_node=1 open_instruct/dpo.py \
    --model.model_name_or_path Qwen/Qwen3-0.6B \
    --chat_template_name qwen \
    --training.max_seq_length 1024 \
    --training.per_device_train_batch_size 1 \
    --training.gradient_accumulation_steps 4 \
    --training.learning_rate 5e-07 \
    --training.lr_scheduler_type linear \
    --training.warmup_ratio 0.1 \
    --training.weight_decay 0.0 \
    --training.num_epochs 3 \
    --checkpoint.output_dir output/dpo_qwen_debug/ \
    --logging.logging_steps 1 \
    --dataset.mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 100 \
    --dataset.skip_cache \
    --tracking.seed 123 \
    --model.use_flash_attn \
    --training.gradient_checkpointing \
    --logging.with_tracking
