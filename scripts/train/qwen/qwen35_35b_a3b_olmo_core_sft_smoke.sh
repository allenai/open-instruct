#!/bin/bash
set -euo pipefail

BEAKER_IMAGE="${1:?Pass the Qwen 3.5 Open Instruct Beaker image as the first argument}"
PROJECT_ROOT="${PROJECT_ROOT:-/weka/oe-adapt-default/jacobm/olmoe3/post-training}"
EXP_NAME="${EXP_NAME:-qwen35-35b-a3b-dolci-olmo-core-sft-smoke}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}-$(date +%Y%m%d-%H%M%S)}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/qwen3.5-35b-a3b-base-olmo}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/datasets/Dolci-Think-SFT-32B/qwen3.5-35b-a3b-olmo_thinker/smoke-1k-4096-cpu}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/${RUN_NAME}}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2}"
COMPILE_MODEL="${COMPILE_MODEL:-false}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
CHECKPOINTING_ENABLED="${CHECKPOINTING_ENABLED:-true}"
ACTIVATION_MEMORY_BUDGET="${ACTIVATION_MEMORY_BUDGET:-1.0}"
ACTIVATION_CHECKPOINTING_MODE="${ACTIVATION_CHECKPOINTING_MODE:-budget}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_NODES="${NUM_NODES:-1}"
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS))
EP_DEGREE="${EP_DEGREE:-${TOTAL_GPUS}}"
CP_DEGREE="${CP_DEGREE:-}"
CP_STRATEGY="${CP_STRATEGY:-ulysses}"
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/OLMo-3-moe-experiments}"

cp_args=()
if [[ -n "$CP_DEGREE" ]]; then
    cp_args+=(--cp_degree "$CP_DEGREE" --cp_strategy "$CP_STRATEGY")
fi

torchrun_args=(--nproc_per_node="$NUM_GPUS")
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
    --workspace "$BEAKER_WORKSPACE" \
    --priority urgent \
    --timeout 6h \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --non_resumable \
    --no_auto_dataset_cache \
    --num_nodes "$NUM_NODES" \
    --gpus "$NUM_GPUS" \
    --env OLMO_SHARED_FS=1 \
    --env OLMO_DDP_INIT_SYNC=0 \
    --env OLMO_EP_MP_HIGH_PRIORITY_GROUP=0 \
    --env NVSHMEM_ENABLE_NIC_PE_MAPPING=0 \
    --env NVSHMEM_HCA_LIST= \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env PYTHONPATH="${PROJECT_ROOT}/OLMo-core/src:${PROJECT_ROOT}/open-instruct" \
    --env DOCUMENT_BOUNDARY_START_TOKEN='<|im_start|>' \
    -- torchrun \
        "${torchrun_args[@]}" \
        "${PROJECT_ROOT}/open-instruct/open_instruct/olmo_core_finetune.py" \
        --run_name "$RUN_NAME" \
        --exp_name "$EXP_NAME" \
        --model_name_or_path "$MODEL_PATH" \
        --config_name Qwen/Qwen3.5-35B-A3B-Base \
        --tokenizer_name_or_path Qwen/Qwen3.5-35B-A3B \
        --pretokenized_dataset_path "$DATASET_PATH" \
        --ensure_terminal_eos_after_truncation true \
        --document_boundary_start_token \$DOCUMENT_BOUNDARY_START_TOKEN \
        --output_dir "$OUTPUT_DIR" \
        --attn_implementation flash_4 \
        --max_seq_length "$MAX_SEQ_LENGTH" \
        --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
        --max_train_steps "$MAX_TRAIN_STEPS" \
        --num_epochs 1 \
        --learning_rate "$LEARNING_RATE" \
        --lr_scheduler_type constant \
        --warmup_ratio 0.0 \
        --weight_decay 0.0 \
        --max_grad_norm 1.0 \
        --moe_expert_parallel_degree "$EP_DEGREE" \
        --moe_expert_parallel_path sync_1d \
        --moe_recompute_each_block false \
        --moe_checkpoint_block_internals true \
        --compile_model "$COMPILE_MODEL" \
        --activation_memory_budget "$ACTIVATION_MEMORY_BUDGET" \
        --activation_checkpointing_mode "$ACTIVATION_CHECKPOINTING_MODE" \
        "${cp_args[@]}" \
        --checkpointing_enabled "$CHECKPOINTING_ENABLED" \
        --checkpointing_steps 1000 \
        --ephemeral_save_interval 500 \
        --keep_last_n_checkpoints 1 \
        --logging_steps 1 \
        --seed 123 \
        --data_loader_seed 456
