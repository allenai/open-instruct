#!/bin/bash
set -euo pipefail

NUM_NODES="${NUM_NODES:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-6}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
COMPILE_MODEL="${COMPILE_MODEL:-false}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
DATASET_VARIANT="${DATASET_VARIANT:-10k}"
MOE_RECOMPUTE_EACH_BLOCK="${MOE_RECOMPUTE_EACH_BLOCK:-true}"
MOE_CHECKPOINT_BLOCK_INTERNALS="${MOE_CHECKPOINT_BLOCK_INTERNALS:-true}"
MOE_EXPERT_PARALLEL_PATH="${MOE_EXPERT_PARALLEL_PATH:-sync_1d}"
MOE_EXPERT_PARALLEL_CAPACITY_FACTOR="${MOE_EXPERT_PARALLEL_CAPACITY_FACTOR:-1.25}"
BEAKER_IMAGE="${1:?Pass the Open Instruct Beaker image as the first argument}"

EXP_NAME="qwen3-30b-a3b-olmo-core-sft-32k-bench-${DATASET_VARIANT}-${NUM_NODES}n-mb${PER_DEVICE_TRAIN_BATCH_SIZE}-ga${GRADIENT_ACCUMULATION_STEPS}-compile${COMPILE_MODEL}-recompute${MOE_RECOMPUTE_EACH_BLOCK}-internal${MOE_CHECKPOINT_BLOCK_INTERNALS}-${MOE_EXPERT_PARALLEL_PATH}-cap${MOE_EXPERT_PARALLEL_CAPACITY_FACTOR}"
RUN_NAME="${EXP_NAME}-$(date +%Y%m%d-%H%M%S)"
PROJECT_ROOT="/weka/oe-adapt-default/jacobm/olmoe3/post-training"
MODEL_PATH="${PROJECT_ROOT}/checkpoints/qwen3-30b-a3b-base-olmo"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/datasets/Dolci-Think-SFT-32B/qwen3-30b-a3b-olmo_thinker-terminal-eos-v2/${DATASET_VARIANT}}"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/benchmarks/${RUN_NAME}"

torchrun_args=(--nproc_per_node=8)
if [[ "$NUM_NODES" == "1" ]]; then
    torchrun_args+=(--standalone)
else
    torchrun_args+=(
        --nnodes="$NUM_NODES"
        --node_rank=\$BEAKER_REPLICA_RANK
        --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME
        --master_port=29400
    )
fi

uv run python mason.py \
    --task_name "$EXP_NAME" \
    --description "$RUN_NAME" \
    --cluster ai2/holmes \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --non_resumable \
    --no_auto_dataset_cache \
    --num_nodes "$NUM_NODES" \
    --gpus 8 \
    --env OLMO_SHARED_FS=1 \
    --env OLMO_DDP_INIT_SYNC=0 \
    --env OLMO_EP_MP_HIGH_PRIORITY_GROUP=0 \
    --env NVSHMEM_ENABLE_NIC_PE_MAPPING=0 \
    --env NVSHMEM_HCA_LIST= \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env PYTHONPATH="${PROJECT_ROOT}/open-instruct" \
    --env DOCUMENT_BOUNDARY_START_TOKEN='<|im_start|>' \
    -- torchrun \
    "${torchrun_args[@]}" \
    "${PROJECT_ROOT}/open-instruct/open_instruct/olmo_core_finetune.py" \
    --run_name "$RUN_NAME" \
    --exp_name "$EXP_NAME" \
    --model_name_or_path "$MODEL_PATH" \
    --config_name Qwen/Qwen3-30B-A3B-Base \
    --tokenizer_name_or_path Qwen/Qwen3-30B-A3B \
    --pretokenized_dataset_path "$DATASET_PATH" \
    --ensure_terminal_eos_after_truncation true \
    --document_boundary_start_token \$DOCUMENT_BOUNDARY_START_TOKEN \
    --output_dir "$OUTPUT_DIR" \
    --attn_implementation flash_4 \
    --max_seq_length 32768 \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --max_train_steps "$MAX_TRAIN_STEPS" \
    --num_epochs 1 \
    --learning_rate 4e-5 \
    --lr_scheduler_type constant \
    --warmup_ratio 0.0 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    --moe_expert_parallel_degree 8 \
    --moe_expert_parallel_path "$MOE_EXPERT_PARALLEL_PATH" \
    --moe_expert_parallel_capacity_factor "$MOE_EXPERT_PARALLEL_CAPACITY_FACTOR" \
    --moe_recompute_each_block "$MOE_RECOMPUTE_EACH_BLOCK" \
    --moe_checkpoint_block_internals "$MOE_CHECKPOINT_BLOCK_INTERNALS" \
    --compile_model "$COMPILE_MODEL" \
    --activation_memory_budget 1.0 \
    --checkpointing_enabled false \
    --checkpointing_steps 1000000 \
    --logging_steps "$LOGGING_STEPS" \
    --seed 123
