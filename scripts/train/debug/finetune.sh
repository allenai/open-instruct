#!/bin/bash
# SFT debug script using OLMo-core training infrastructure.
#
# This script first creates a numpy dataset from HuggingFace data using
# --cache_dataset_only mode, then runs training on the cached data.
#
# Usage:
#   # Run on Beaker:
#   ./scripts/train/debug/finetune.sh <beaker_image>

RUN_NAME="debug-sft-olmo3-7b"
CHECKPOINT="/weka/oe-training-default/ai2-llm/checkpoints/OLMo3-midtraining/anneal-round5-100B-olmo3_7b-anneal-decon-12T-00bb6023/step47684"
CLUSTER="ai2/jupiter"
SEQ_LEN=4096
NUM_NODES=1
GLOBAL_BATCH_SIZE=$((64 * SEQ_LEN))
BUDGET="ai2/oe-adapt"
WORKSPACE="ai2/open-instruct-dev"
NUM_EPOCHS=1
MODEL_CONFIG="olmo3_7B"

if [ -n "$1" ]; then
    BEAKER_IMAGE="$1"
    echo "Using Beaker image: $BEAKER_IMAGE"

    uv run python mason.py \
        --cluster "$CLUSTER" \
        --workspace "$WORKSPACE" \
        --priority urgent \
        --image "$BEAKER_IMAGE" \
        --description "OLMo-core SFT debug job - cache dataset and train." \
        --pure_docker_mode \
        --preemptible \
        --num_nodes "$NUM_NODES" \
        --budget "$BUDGET" \
        --gpus 8 \
        --non_resumable \
        -- \
        bash -c '
set -e
OUTPUT_DIR="/weka/oe-adapt-default/allennlp/deletable_checkpoint/$(whoami)/debug-sft-data"

echo "Step 1: Caching dataset to numpy format..."
python open_instruct/finetune.py cache_dataset_only \
    --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture 0.01 \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_length '"$SEQ_LEN"' \
    --tokenizer_name_or_path allenai/OLMo-2-1124-7B

echo "Step 2: Running training..."
torchrun --nproc_per_node=8 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 open_instruct/finetune.py train \
    '"$RUN_NAME"' \
    '"$CHECKPOINT"' \
    '"$CLUSTER"' \
    --dataset_path "$OUTPUT_DIR" \
    --seq_len '"$SEQ_LEN"' \
    --num_nodes '"$NUM_NODES"' \
    --global_batch_size '"$GLOBAL_BATCH_SIZE"' \
    --budget '"$BUDGET"' \
    --workspace '"$WORKSPACE"' \
    --num_epochs '"$NUM_EPOCHS"' \
    --model_config '"$MODEL_CONFIG"' \
    --wandb_project open_instruct_internal \
    --wandb_entity ai2-llm
'
else
    echo "Running locally..."
    echo "Note: Local execution requires a GPU with CUDA support."
    OUTPUT_DIR="output/debug-sft-data"

    echo "Step 1: Caching dataset to numpy format..."
    uv run python open_instruct/finetune.py cache_dataset_only \
        --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture 0.01 \
        --output_dir "$OUTPUT_DIR" \
        --max_seq_length "$SEQ_LEN" \
        --tokenizer_name_or_path allenai/OLMo-2-1124-7B

    echo "Step 2: Running training..."
    uv run torchrun --nproc_per_node=8 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 open_instruct/finetune.py train \
        "$RUN_NAME" \
        "$CHECKPOINT" \
        "$CLUSTER" \
        --dataset_path "$OUTPUT_DIR" \
        --seq_len "$SEQ_LEN" \
        --num_nodes "$NUM_NODES" \
        --global_batch_size "$GLOBAL_BATCH_SIZE" \
        --num_epochs "$NUM_EPOCHS" \
        --model_config "$MODEL_CONFIG"
fi
