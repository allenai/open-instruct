#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
OUTPUT_DIR="/weka/oe-adapt-default/allennlp/deletable_checkpoint/dpo_cache_multinode_ckpt_test"

COMMON_ARGS="accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
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
    --num_epochs 3 \
    --output_dir ${OUTPUT_DIR} \
    --logging_steps 1 \
    --mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 100 \
    --add_bos \
    --chat_template_name olmo \
    --seed 123 \
    --activation_memory_budget 0.5 \
    --checkpointing_steps 3 \
    --keep_last_n_checkpoints 2 \
    --with_tracking \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false"

MASON_ARGS="--cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image $BEAKER_IMAGE \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 2 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --non_resumable \
    --gpus 8"

# Job 1: Train 10 steps with checkpointing every 3 steps, keeping the last 2.
# Expected checkpoints at steps 3, 6, 9. After cleanup, step_3 is removed; step_6 and step_9 remain.
echo "=== Job 1: Training with checkpointing (10 steps) ==="
uv run python mason.py \
    ${MASON_ARGS} \
    --description "DPO cache checkpoint test: Job 1 (train 10 steps)" \
    -- ${COMMON_ARGS} \
    --exp_name dpo-cache-multinode-ckpt-test-job1 \
    --max_train_steps 10

# Job 2: Resume from the last checkpoint (step_9) and train to step 15.
# The resume logic in get_last_checkpoint_path finds step_9 in OUTPUT_DIR.
echo "=== Job 2: Resuming from checkpoint (to 15 steps) ==="
uv run python mason.py \
    ${MASON_ARGS} \
    --description "DPO cache checkpoint test: Job 2 (resume to 15 steps)" \
    -- ${COMMON_ARGS} \
    --exp_name dpo-cache-multinode-ckpt-test-job2 \
    --max_train_steps 15
