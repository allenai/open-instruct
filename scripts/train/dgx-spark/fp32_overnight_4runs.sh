#!/bin/bash
# =============================================================================
# FP32 LM Head Comparison: 4 Runs (2 configs × 2 fp32 modes)
# =============================================================================
#
# Runs 4 experiments overnight:
#   1. grpo_qwen_gsm8k.sh (long response) + no fp32
#   2. grpo_qwen_gsm8k.sh (long response) + fp32 permanent
#   3. grpo_qwen_gsm8k_short.sh (short response) + no fp32
#   4. grpo_qwen_gsm8k_short.sh (short response) + fp32 permanent
#
# Usage:
#   ./scripts/train/dgx-spark/fp32_overnight_4runs.sh
#
# =============================================================================

set -e
cd "$(dirname "$0")/../../.."

echo "=============================================="
echo "FP32 LM Head Comparison - 4 Overnight Runs"
echo "=============================================="
echo "Start time: $(date)"
echo ""

cleanup_between_runs() {
    echo "Cleaning up between runs..."
    pkill -9 -f "ray::" 2>/dev/null || true
    pkill -9 -f "grpo_fast" 2>/dev/null || true
    sleep 10
    FREE_MEM_GB=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
    echo "Available memory: ${FREE_MEM_GB}GB"
}

# =============================================================================
# Run 1: Long response config + no fp32
# =============================================================================
echo ""
echo "=============================================="
echo "Run 1/4: Long response + NO FP32"
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
echo "Run 2/4: Long response + FP32 PERMANENT"
echo "Time: $(date)"
echo "=============================================="
cleanup_between_runs

FP32_LM_HEAD=1 FP32_PERMANENT=1 WITH_TRACKING=1 \
    ./scripts/train/dgx-spark/grpo_qwen_gsm8k.sh

# =============================================================================
# Run 3: Short response config + no fp32
# =============================================================================
echo ""
echo "=============================================="
echo "Run 3/4: Short response + NO FP32"
echo "Time: $(date)"
echo "=============================================="
cleanup_between_runs

FP32_MODE=none \
    ./scripts/train/dgx-spark/grpo_qwen_gsm8k_short.sh

# =============================================================================
# Run 4: Short response config + fp32 permanent
# =============================================================================
echo ""
echo "=============================================="
echo "Run 4/4: Short response + FP32 PERMANENT"
echo "Time: $(date)"
echo "=============================================="
cleanup_between_runs

FP32_MODE=permanent \
    ./scripts/train/dgx-spark/grpo_qwen_gsm8k_short.sh

echo ""
echo "=============================================="
echo "All 4 runs completed at: $(date)"
echo "=============================================="
