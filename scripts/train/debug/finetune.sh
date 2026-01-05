#!/bin/bash
# SFT debug script using OLMo-core training infrastructure.
#
# Usage:
#   # Run on Beaker:
#   ./scripts/train/debug/finetune.sh <beaker_image>

RUN_NAME="debug-sft-qwen3"
CHECKPOINT="Qwen/Qwen3-0.6B"
CLUSTER="ai2/jupiter"
SEQ_LEN=1024
NUM_NODES=1
GLOBAL_BATCH_SIZE=$((64 * SEQ_LEN))
BUDGET="ai2/oe-adapt"
WORKSPACE="ai2/open-instruct-dev"
NUM_EPOCHS=2

LAUNCH_CMD="torchrun --nproc_per_node=8 open_instruct/finetune.py train \
    $RUN_NAME \
    $CHECKPOINT \
    $CLUSTER \
    --seq_len $SEQ_LEN \
    --num_nodes $NUM_NODES \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --budget $BUDGET \
    --workspace $WORKSPACE \
    --num_epochs $NUM_EPOCHS \
    --dataset_mixer_list allenai/tulu-3-sft-personas-algebra 1.0 \
    --wandb_project open_instruct_internal \
    --wandb_entity ai2-llm"

if [ -n "$1" ]; then
    BEAKER_IMAGE="$1"
    echo "Using Beaker image: $BEAKER_IMAGE"

    uv run python mason.py \
        --cluster "$CLUSTER" \
        --workspace "$WORKSPACE" \
        --priority normal \
        --image "$BEAKER_IMAGE" \
        --description "Single GPU OLMo-core SFT debug job." \
        --pure_docker_mode \
        --preemptible \
        --num_nodes "$NUM_NODES" \
        --budget "$BUDGET" \
        --gpus 8 \
        --non_resumable \
        -- \
        $LAUNCH_CMD
else
    echo "Running locally..."
    echo "Note: Local execution requires a GPU with CUDA support."
    uv run $LAUNCH_CMD
fi
