#!/bin/bash
# Pure olmo-core SFT, Qwen3-0.6B, single GPU, 3 steps.
# Pair: scripts/train/debug/oc_sft_qwen3_openinstruct.sh (same hyperparams).

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

DATASET_PATH=/weka/oe-adapt-default/nathanl/dataset/olmo3-32b-instruct-sft-1114

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Qwen3-0.6B pure olmo-core SFT match test (3 steps)" \
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
    scripts/train/debug/olmo_core_reference_sft.py \
    --dataset_path "$DATASET_PATH" \
    --save_folder \$CHECKPOINT_OUTPUT_DIR \
    --seq_len 1024 \
    --rank_microbatch_size_tokens 1024 \
    --global_batch_size_tokens 4096 \
    --learning_rate 5e-6 \
    --warmup_fraction 0.03 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    --max_steps 4 \
    --init_seed 33333 \
    --data_loader_seed 34521
