#!/bin/bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-qwen3-30b-a3b-dolci-think-olmo-core-sft-smoke}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}-$(date +%Y%m%d-%H%M%S)}"
BEAKER_IMAGE="${1:?Pass the Open Instruct Beaker image as the first argument}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to an OLMo-core Qwen3-MoE checkpoint}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to a pretokenized OLMo-core SFT dataset}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for OLMo-core checkpoints}"

uv run python mason.py \
    --task_name "$EXP_NAME" \
    --description "$RUN_NAME" \
    --cluster ai2/holmes \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --non_resumable \
    --preemptible \
    --no_auto_dataset_cache \
    --num_nodes 1 \
    --gpus 8 \
    --env OLMO_SHARED_FS=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env DOCUMENT_BOUNDARY_START_TOKEN='<|im_start|>' \
    -- torchrun \
    --standalone \
    --nproc_per_node=8 \
    open_instruct/olmo_core_finetune.py \
    --run_name "$RUN_NAME" \
    --exp_name "$EXP_NAME" \
    --model_name_or_path "$MODEL_PATH" \
    --config_name Qwen/Qwen3-30B-A3B-Base \
    --tokenizer_name_or_path Qwen/Qwen3-30B-A3B \
    --pretokenized_dataset_path "$DATASET_PATH" \
    --ensure_terminal_eos_after_truncation true \
    --document_boundary_start_token \$DOCUMENT_BOUNDARY_START_TOKEN \
    --output_dir "$OUTPUT_DIR" \
    --attn_implementation torch \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 1 \
    --num_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.0 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    --moe_expert_parallel_degree 8 \
    --moe_expert_parallel_path sync_1d \
    --moe_recompute_each_block true \
    --compile_model false \
    --activation_memory_budget 1.0 \
    --checkpointing_steps 1000 \
    --ephemeral_save_interval 500 \
    --logging_steps 1 \
    --seed 123
