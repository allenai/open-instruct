#!/bin/bash
# =============================================================================
# FP32 LM Head Comparison: 2 Short Runs
# =============================================================================
#
# Runs 2 experiments with short response config (more conservative memory):
#   1. grpo_qwen_gsm8k_short.sh + no fp32
#   2. grpo_qwen_gsm8k_short.sh + fp32 permanent
#
# Usage:
#   ./scripts/train/dgx-spark/fp32_short_2runs.sh
#
# =============================================================================

set -e
cd "$(dirname "$0")/../../.."

# Ensure unbuffered output for real-time logging
export PYTHONUNBUFFERED=1

echo "=============================================="
echo "FP32 LM Head Comparison - 2 Short Runs"
echo "=============================================="
echo "Start time: $(date)"
echo ""

cleanup_between_runs() {
    echo "Cleaning up between runs..."
    pkill -9 -f "ray::" 2>/dev/null || true
    pkill -9 -f "grpo_fast" 2>/dev/null || true
    pkill -9 -f "VLLM" 2>/dev/null || true
    sleep 15
    FREE_MEM_GB=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
    echo "Available memory: ${FREE_MEM_GB}GB"
}

# =============================================================================
# Run 1: Short response config + no fp32
# =============================================================================
echo ""
echo "=============================================="
echo "Run 1/2: Short response + NO FP32"
echo "Time: $(date)"
echo "=============================================="
cleanup_between_runs

FP32_MODE=none WITH_TRACKING=1 \
    ./scripts/train/dgx-spark/grpo_qwen_gsm8k_short.sh

# =============================================================================
# Run 2: Short response config + fp32 permanent
# =============================================================================
echo ""
echo "=============================================="
echo "Run 2/2: Short response + FP32 PERMANENT"
echo "Time: $(date)"
echo "=============================================="
cleanup_between_runs

FP32_MODE=permanent WITH_TRACKING=1 \
    ./scripts/train/dgx-spark/grpo_qwen_gsm8k_short.sh

echo ""
echo "=============================================="
echo "Both runs completed at: $(date)"
echo "=============================================="
