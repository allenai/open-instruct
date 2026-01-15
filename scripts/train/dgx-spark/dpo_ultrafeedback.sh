#!/bin/bash
# DPO training on DGX Spark (single GPU, NVIDIA GB10)
# Based on scripts/train/debug/dpo.sh
#
# Usage: ./scripts/train/dgx-spark/dpo_ultrafeedback.sh [model_path]
#
# Arguments:
#   model_path: Path to SFT model (default: searches for latest SFT output)
#
# This trains on argilla/ultrafeedback-binarized-preferences-cleaned (10K subsample).
# DGX Spark notes:
#   - Use --use_flash_attn false (SDPA is faster on Blackwell)
#   - DPO needs ~2x memory of SFT (policy + reference model)
#   - Batch 16 @ 1024 ctx is conservative for unified memory
#
# OOM PROTECTION:
#   - DGX Spark has unified memory (119GB shared CPU/GPU)
#   - OOM can freeze the entire system, not just kill the process
#   - This script includes pre-flight checks and conservative settings

set -e  # Exit on error

cd "$(dirname "$0")/../../.."
export PATH="$HOME/.local/bin:$PATH"

# =============================================================================
# OOM Protection for DGX Spark Unified Memory
# =============================================================================

# DPO needs more memory than SFT (policy + reference model)
MIN_FREE_MEM_GB=60

# Check for leftover GPU processes that might be holding memory
cleanup_gpu_processes() {
    echo "Checking for leftover GPU processes..."
    if pgrep -f "ray::" > /dev/null || pgrep -f "VLLM" > /dev/null; then
        echo "WARNING: Found leftover ray/vLLM processes. Cleaning up..."
        pkill -9 -f "ray::" 2>/dev/null || true
        pkill -9 -f "VLLM" 2>/dev/null || true
        ray stop --force 2>/dev/null || true
        echo "Waiting 10s for memory to be released..."
        sleep 10
    fi
}

# Check available memory before starting
check_memory() {
    local free_mem_kb=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    local free_mem_gb=$((free_mem_kb / 1024 / 1024))
    echo "Available memory: ${free_mem_gb}GB (minimum required: ${MIN_FREE_MEM_GB}GB)"

    if [ "$free_mem_gb" -lt "$MIN_FREE_MEM_GB" ]; then
        echo "ERROR: Not enough free memory. Current: ${free_mem_gb}GB, Required: ${MIN_FREE_MEM_GB}GB"
        echo "Run: pkill -9 -f 'ray::' && pkill -9 -f 'VLLM' && sleep 10"
        exit 1
    fi
}

# Set PyTorch memory management for unified memory
setup_pytorch_memory() {
    # Limit memory fragmentation - important for unified memory
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

    # Enable garbage collection more aggressively
    export PYTORCH_NO_CUDA_MEMORY_CACHING=0
}

# Trap to cleanup on script exit/error
cleanup_on_exit() {
    echo "Cleaning up..."
    # Kill any background processes we might have started
    jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup_on_exit EXIT INT TERM

# =============================================================================
# Pre-flight checks
# =============================================================================

echo "=== DGX Spark DPO Training ==="

# Find SFT model - use argument or search for latest output
if [ -n "$1" ]; then
    MODEL_PATH="$1"
else
    # Find latest SFT output directory
    MODEL_PATH=$(ls -td /tmp/sft_qwen3_no_robots/spark_sft_qwen3_no_robots__* 2>/dev/null | head -1)
    if [ -z "$MODEL_PATH" ]; then
        echo "Error: No SFT model found. Run sft_no_robots.sh first or provide model path."
        exit 1
    fi
fi

echo "Model: $MODEL_PATH"
echo "Dataset: argilla/ultrafeedback-binarized-preferences-cleaned (10K)"
echo ""

cleanup_gpu_processes
check_memory
setup_pytorch_memory

echo "Memory check passed. Starting training..."
echo ""

# =============================================================================
# Training command
# =============================================================================

# Conservative settings for unified memory:
# - Batch size 16 (DPO needs ~2x memory of SFT)
# - Gradient checkpointing enabled
# - 1024 context length

LAUNCH_CMD="python -m accelerate.commands.launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    open_instruct/dpo_tune_cache.py \
    --exp_name spark_dpo_qwen3_ultrafeedback \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name $MODEL_PATH \
    --use_flash_attn false \
    --max_seq_length 1024 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-7 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --dataset_mixer_list argilla/ultrafeedback-binarized-preferences-cleaned 10000 \
    --add_bos \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --seed 42 \
    --output_dir /tmp/dpo_qwen3_ultrafeedback \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --try_launch_beaker_eval_jobs false"

echo "Running locally on DGX Spark..."
echo "Command: uv run $LAUNCH_CMD"
echo ""

# Run with timeout to prevent infinite hangs
# 4 hours should be enough for this training run
timeout --signal=KILL 14400 uv run $LAUNCH_CMD

echo ""
echo "=== Training complete ==="
