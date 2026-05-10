#!/bin/bash

# SFT for Qwen3.5-9B on a schema-normalized copy of rl-rag-2/sft_ablations_bc_only_v1.
# Launch with:
#   ./scripts/train/build_image_and_launch.sh scripts/train/qwen/sft_qwen35_9b_rl_rag_2_bc_only_v1.sh

set -euo pipefail

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
DATASET="${DATASET:-hamishivi/sft_ablations_bc_only_v1_sanitized}"
EXP_NAME="${EXP_NAME:-sft_qwen35_9b_rl_rag_2_bc_only_v1}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-32768}"
NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
SEQUENCE_PARALLEL_SIZE="${SEQUENCE_PARALLEL_SIZE:-2}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"

WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
if ((WORLD_SIZE % SEQUENCE_PARALLEL_SIZE != 0)); then
    echo "WORLD_SIZE=$WORLD_SIZE must be divisible by SEQUENCE_PARALLEL_SIZE=$SEQUENCE_PARALLEL_SIZE" >&2
    exit 1
fi

DATA_PARALLEL_SIZE=$((WORLD_SIZE / SEQUENCE_PARALLEL_SIZE))
MICRO_BATCH_SIZE=$((DATA_PARALLEL_SIZE * PER_DEVICE_TRAIN_BATCH_SIZE))
if ((TOTAL_BATCH_SIZE % MICRO_BATCH_SIZE != 0)); then
    echo "TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE must be divisible by DATA_PARALLEL_SIZE * PER_DEVICE_TRAIN_BATCH_SIZE=$MICRO_BATCH_SIZE" >&2
    exit 1
fi

GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / MICRO_BATCH_SIZE))
echo "Total batch size: $TOTAL_BATCH_SIZE = DP($DATA_PARALLEL_SIZE) * per-device($PER_DEVICE_TRAIN_BATCH_SIZE) * grad-acc($GRADIENT_ACCUMULATION_STEPS), with SP=$SEQUENCE_PARALLEL_SIZE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/dr-tulu-ablations \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes "$NUM_NODES" \
    --env BEAKER_ALLOW_SUBCONTAINERS=1 \
    --env BEAKER_SKIP_DOCKER_SOCKET=1 \
    --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
    --budget ai2/oe-adapt \
    --gpus "$GPUS_PER_NODE" \
    --no_auto_dataset_cache \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes "$GPUS_PER_NODE" \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name "$EXP_NAME" \
    --model_name_or_path "$MODEL" \
    --tokenizer_name "$MODEL" \
    --use_liger_kernel \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --sequence_parallel_size "$SEQUENCE_PARALLEL_SIZE" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --dataset_mixer_list "$DATASET" 1.0 \
    --dataset_mixer_list_splits train \
    --add_bos \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --seed 42 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
