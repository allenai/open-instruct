#!/bin/bash
# SFT debug script using OLMo-core training infrastructure.
#
# Usage:
#   # Run on Beaker:
#   ./scripts/train/debug/finetune.sh <beaker_image>

RUN_NAME="debug-sft-olmo2-7b"
CHECKPOINT="s3://ai2-llm/checkpoints/OLMo-2/Olmo-2-1124-7B/step556000-unsharded"
CLUSTER="ai2/jupiter"
DATASET_PATH="/weka/oe-training-default/ai2-llm/preprocessed/tulu-sft/tulu-3-sft-olmo-2-mixture-4096"
SEQ_LEN=4096
NUM_NODES=1
GLOBAL_BATCH_SIZE=$((64 * SEQ_LEN))
BUDGET="ai2/oe-adapt"
WORKSPACE="ai2/open-instruct-dev"
NUM_EPOCHS=1

if [ -n "$1" ]; then
    BEAKER_IMAGE="$1"
    echo "Using Beaker image: $BEAKER_IMAGE"

    uv run python mason.py \
        --cluster "$CLUSTER" \
        --workspace "$WORKSPACE" \
        --priority urgent \
        --image "$BEAKER_IMAGE" \
        --description "OLMo-core SFT debug job." \
        --pure_docker_mode \
        --preemptible \
        --num_nodes "$NUM_NODES" \
        --budget "$BUDGET" \
        --gpus 8 \
        --non_resumable \
        --no_auto_dataset_cache \
        -- \
        torchrun --nproc_per_node=8 open_instruct/finetune.py train \
            "$RUN_NAME" \
            "$CHECKPOINT" \
            "$CLUSTER" \
            --dataset_path "$DATASET_PATH" \
            --seq_len "$SEQ_LEN" \
            --num_nodes "$NUM_NODES" \
            --global_batch_size "$GLOBAL_BATCH_SIZE" \
            --budget "$BUDGET" \
            --workspace "$WORKSPACE" \
            --num_epochs "$NUM_EPOCHS" \
            --wandb_project open_instruct_internal \
            --wandb_entity ai2-llm
else
    echo "Running locally..."
    echo "Note: Local execution requires a GPU with CUDA support."
    uv run torchrun --nproc_per_node=8 open_instruct/finetune.py train \
        "$RUN_NAME" \
        "$CHECKPOINT" \
        "$CLUSTER" \
        --dataset_path "$DATASET_PATH" \
        --seq_len "$SEQ_LEN" \
        --num_nodes "$NUM_NODES" \
        --global_batch_size "$GLOBAL_BATCH_SIZE" \
        --num_epochs "$NUM_EPOCHS"
fi
