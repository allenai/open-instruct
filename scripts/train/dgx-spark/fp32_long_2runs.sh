#!/bin/bash
# =============================================================================
# FP32 LM Head Comparison: 2 Long Runs
# =============================================================================
#
# Runs 2 experiments with long response config:
#   1. grpo_qwen_gsm8k.sh + no fp32
#   2. grpo_qwen_gsm8k.sh + fp32 permanent
#
# Usage:
#   ./scripts/train/dgx-spark/fp32_long_2runs.sh
#
# =============================================================================

set -e
cd "$(dirname "$0")/../../.."

# Ensure unbuffered output for real-time logging
export PYTHONUNBUFFERED=1

echo "=============================================="
echo "FP32 LM Head Comparison - 2 Long Runs"
echo "=============================================="
echo "Start time: $(date)"
echo ""

cleanup_between_runs() {
    echo "Cleaning up thoroughly between runs..."
    pkill -9 -f "ray::" 2>/dev/null || true
    pkill -9 -f "grpo_fast" 2>/dev/null || true
    pkill -9 -f "VLLM" 2>/dev/null || true
    pkill -9 -f "python.*open_instruct" 2>/dev/null || true

    # Wait for GPU memory to be released
    echo "Waiting for GPU memory release..."
    sleep 20

    FREE_MEM_GB=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
    echo "Available memory: ${FREE_MEM_GB}GB"

    if [ "$FREE_MEM_GB" -lt 80 ]; then
        echo "Memory still low, waiting longer..."
        sleep 30
        FREE_MEM_GB=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
        echo "Available memory after extra wait: ${FREE_MEM_GB}GB"
    fi
}

# =============================================================================
# Run 1: Long response config + no fp32
# =============================================================================
echo ""
echo "=============================================="
echo "Run 1/2: Long response + NO FP32"
echo "Time: $(date)"
echo "=============================================="
cleanup_between_runs

FP32_LM_HEAD=0 WITH_TRACKING=1 \
    ./scripts/train/dgx-spark/grpo_qwen_gsm8k.sh

# =============================================================================
# Run 2: Long response config + fp32 permanent
# =============================================================================
echo ""
echo "=============================================="
echo "Run 2/2: Long response + FP32 PERMANENT"
echo "Time: $(date)"
echo "=============================================="
cleanup_between_runs

FP32_LM_HEAD=1 FP32_PERMANENT=1 WITH_TRACKING=1 \
    ./scripts/train/dgx-spark/grpo_qwen_gsm8k.sh

echo ""
echo "=============================================="
echo "Both runs completed at: $(date)"
echo "=============================================="
