#!/bin/bash
# Backend A/B: SFT on DeepSpeed (finetune.py). Pair: ab_sft_olmocore.sh.
# See docs/superpowers/specs/2026-08-05-backend-consolidation-design.md Part 1.
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Backend A/B: SFT DeepSpeed (finetune.py), OLMo-2-7B, 2 nodes." \
    --pure_docker_mode \
    --preemptible \
    --max_retries 0 \
    --num_nodes 2 \
    --gpus 8 \
    --non_resumable \
    --no_auto_dataset_cache \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name ab_sft_deepspeed \
    --model_name_or_path allenai/OLMo-2-1124-7B \
    --tokenizer_name allenai/OLMo-2-1124-7B \
    --add_bos \
    --chat_template_name tulu \
    --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 60000 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --max_train_steps 150 \
    --logging_steps 1 \
    --seed 42 \
    --report_to wandb \
    --with_tracking \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
