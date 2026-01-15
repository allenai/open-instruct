#!/bin/bash
# Memory profiling script for DPO on DGX Spark
#
# Usage: ./scripts/train/dgx-spark/memory_profile_dpo.sh <batch_size> <grad_accum> <seq_len> [model]
#
# Example:
#   ./scripts/train/dgx-spark/memory_profile_dpo.sh 1 1 1024
#   ./scripts/train/dgx-spark/memory_profile_dpo.sh 4 4 1024 Qwen/Qwen2.5-1.5B
#
# DPO uses ~1.8x memory of SFT (policy + frozen reference model)
# This runs a SHORT training job (30 steps) to measure peak memory usage.

set -e

cd "$(dirname "$0")/../../.."
export PATH="$HOME/.local/bin:$PATH"

# Parse arguments
BATCH_SIZE=${1:-1}
GRAD_ACCUM=${2:-1}
SEQ_LEN=${3:-1024}
MODEL=${4:-"Qwen/Qwen3-0.6B"}

# Extract model short name for experiment naming
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')

echo "========================================"
echo "DGX Spark Memory Profiling - DPO"
echo "========================================"
echo "Model:           $MODEL"
echo "Batch size:      $BATCH_SIZE"
echo "Grad accum:      $GRAD_ACCUM"
echo "Seq length:      $SEQ_LEN"
echo "Total batch:     $((BATCH_SIZE * GRAD_ACCUM))"
echo "========================================"
echo ""
echo "NOTE: DPO uses ~1.8x memory of SFT (policy + reference)"
echo ""

# Pre-flight memory check
echo "=== Pre-flight Memory Check ==="
free -h | head -2
echo ""

FREE_MEM_GB=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
echo "Available memory: ${FREE_MEM_GB}GB"

if [ "$FREE_MEM_GB" -lt 40 ]; then
    echo "ERROR: Less than 40GB free. Clean up first:"
    echo "  pkill -9 -f python && sleep 15 && free -h"
    exit 1
fi

# PyTorch memory settings
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# Use local cache path for DGX Spark (not /weka)
export REFERENCE_LOGPROBS_CACHE_PATH="/tmp/reference_logprobs_cache"

# Create unique output dir
OUTPUT_DIR="/tmp/memory_profile_dpo_${MODEL_SHORT}_b${BATCH_SIZE}_g${GRAD_ACCUM}_s${SEQ_LEN}"
rm -rf "$OUTPUT_DIR"

# Memory monitoring file
MEM_LOG="/tmp/mem_monitor_dpo_$$.log"
PEAK_MEM_FILE="/tmp/peak_mem_dpo_$$.txt"
echo "0" > "$PEAK_MEM_FILE"

# Start background memory monitor
(
    while true; do
        MEM_USED_GB=$(awk '/MemTotal/ {total=$2} /MemAvailable/ {avail=$2} END {printf "%.1f", (total-avail)/1024/1024}' /proc/meminfo)
        CURRENT_PEAK=$(cat "$PEAK_MEM_FILE" 2>/dev/null || echo "0")
        if (( $(echo "$MEM_USED_GB > $CURRENT_PEAK" | bc -l) )); then
            echo "$MEM_USED_GB" > "$PEAK_MEM_FILE"
        fi
        echo "$(date '+%H:%M:%S') ${MEM_USED_GB}GB" >> "$MEM_LOG"
        sleep 2
    done
) &
MEM_MONITOR_PID=$!

# Cleanup function
cleanup() {
    kill $MEM_MONITOR_PID 2>/dev/null || true
}
trap cleanup EXIT

echo ""
echo "=== Starting Training (30 steps) ==="
echo "Output dir: $OUTPUT_DIR"
echo "Memory monitor PID: $MEM_MONITOR_PID"
echo ""

# Run SHORT training - just 30 steps to measure memory
# DPO is slower per step so we use fewer steps
uv run python -m accelerate.commands.launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    open_instruct/dpo_tune_cache.py \
    --exp_name "mem_profile_dpo_${MODEL_SHORT}" \
    --model_name_or_path "$MODEL" \
    --tokenizer_name "$MODEL" \
    --use_flash_attn false \
    --max_seq_length "$SEQ_LEN" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate 5e-7 \
    --lr_scheduler_type constant \
    --warmup_ratio 0.0 \
    --weight_decay 0.0 \
    --max_train_steps 30 \
    --logging_steps 10 \
    --dataset_mixer_list argilla/ultrafeedback-binarized-preferences-cleaned 1000 \
    --dataset_skip_cache \
    --add_bos \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --seed 42 \
    --output_dir "$OUTPUT_DIR" \
    --gradient_checkpointing \
    --report_to none \
    --try_launch_beaker_eval_jobs false

# Stop memory monitor and get peak
kill $MEM_MONITOR_PID 2>/dev/null || true
sleep 1

PEAK_MEM=$(cat "$PEAK_MEM_FILE" 2>/dev/null || echo "unknown")

echo ""
echo "========================================"
echo "=== Training Complete ==="
echo "========================================"
echo ""
echo "=== Memory Summary ==="
echo "Peak memory used: ${PEAK_MEM}GB"
free -h | head -2
echo ""
echo "Config: batch=$BATCH_SIZE, grad_accum=$GRAD_ACCUM, seq_len=$SEQ_LEN, model=$MODEL"
echo ""
echo "Memory log saved to: $MEM_LOG"
echo "Last 10 readings:"
tail -10 "$MEM_LOG" 2>/dev/null || echo "(no log)"
echo ""
echo "========================================"
echo "RESULT: DPO batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM seq=$SEQ_LEN peak=${PEAK_MEM}GB model=$MODEL"
echo "========================================"
