#!/bin/bash
# open_instruct/olmo_core_finetune.py SFT, Qwen3-0.6B, single GPU, 3 steps.
# Pair: scripts/train/debug/oc_sft_qwen3_reference.sh (same hyperparams).

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

DATASET_PATH=/weka/oe-adapt-default/nathanl/dataset/olmo3-32b-instruct-sft-1114

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Qwen3-0.6B open-instruct SFT match test (3 steps)" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --gpus 1 \
    --non_resumable \
    --no-host-networking \
    --no_auto_dataset_cache \
    --env OLMO_SHARED_FS=1 \
    -- torchrun --nproc_per_node=1 \
    open_instruct/olmo_core_finetune.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --tokenizer_name_or_path allenai/olmo-3-tokenizer-instruct-dev \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    --max_train_steps 3 \
    --cp_degree 1 \
    --attn_implementation flash_2 \
    --ac_mode selected_modules \
    --ac_modules "blocks.*.feed_forward" \
    --compile_model true \
    --checkpointing_steps 999999 \
    --ephemeral_save_interval 999998 \
    --with_tracking \
    --logging_steps 1 \
    --dataset_path "$DATASET_PATH" \
    --seed 33333 \
    --data_loader_seed 34521 \
    --output_dir \$CHECKPOINT_OUTPUT_DIR
