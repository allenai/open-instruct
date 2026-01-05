#!/bin/bash
# OLMo2-13B SFT training script using OLMo-core infrastructure.
#
# Usage:
#   ./scripts/train/olmo2/finetune_13b.sh
#
# Note: This script requires pre-tokenized numpy data.
# To prepare data, see: python open_instruct/finetune.py cache_dataset_only --help
#
# TODO: Add model_size parameter to finetune.py to support 13B model config

RUN_NAME="olmo2-13b-sft"
CHECKPOINT="s3://ai2-llm/checkpoints/OLMo-2/Olmo-2-1124-13B/step556000-unsharded"
CLUSTER="ai2/jupiter"
DATASET_PATH="s3://ai2-llm/preprocessed/tulu-sft/tulu-3-sft-olmo-2-mixture-4096"
SEQ_LEN=4096
NUM_NODES=8
GLOBAL_BATCH_SIZE=$((64 * SEQ_LEN))
BUDGET="ai2/oe-adapt"
WORKSPACE="ai2/tulu-3-dev"
NUM_EPOCHS=2
LEARNING_RATE=5e-6
WARMUP_RATIO=0.03

python open_instruct/finetune.py launch \
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
    --learning_rate "$LEARNING_RATE" \
    --warmup_ratio "$WARMUP_RATIO" \
    --wandb_project open_instruct_internal \
    --wandb_entity ai2-llm
