#!/bin/bash
# Run full logprobs comparison: vLLM (bf16 + fp32) → HF scoring → plotting
#
# This script runs vLLM in separate processes to avoid memory cleanup issues,
# allowing each run to use ~85% GPU memory instead of ~40%.
#
# Usage:
#   ./scripts/analysis/fp32-lm-head/run_logprobs_comparison.sh
#   ./scripts/analysis/fp32-lm-head/run_logprobs_comparison.sh --model Qwen/Qwen2.5-0.5B --max-tokens 100
#   ./scripts/analysis/fp32-lm-head/run_logprobs_comparison.sh --output-dir ~/dev/logprobs_data/custom/

set -euo pipefail

# Default output directory
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/dev/logprobs_data}"

# Parse --output-dir if provided, pass rest to scripts
SCRIPT_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            SCRIPT_ARGS+=("$1")
            shift
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BF16_OUTPUT="$OUTPUT_DIR/vllm_bf16_${TIMESTAMP}.json"
FP32_OUTPUT="$OUTPUT_DIR/vllm_fp32_${TIMESTAMP}.json"
RESULTS_OUTPUT="$OUTPUT_DIR/results_${TIMESTAMP}.json"

echo "============================================================"
echo "Logprobs Comparison Pipeline"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

# Required for vLLM on aarch64
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export HF_HUB_ENABLE_HF_TRANSFER=1

# Step 1: Generate with vLLM BF16
echo ""
echo "============================================================"
echo "Step 1/4: vLLM BF16 generation"
echo "============================================================"
uv run python scripts/analysis/fp32-lm-head/get_vllm_logprobs.py \
    --mode bf16 \
    --output "$BF16_OUTPUT" \
    "${SCRIPT_ARGS[@]}"

# Step 2: Generate with vLLM FP32 (separate process = clean GPU memory)
echo ""
echo "============================================================"
echo "Step 2/4: vLLM FP32 generation"
echo "============================================================"
uv run python scripts/analysis/fp32-lm-head/get_vllm_logprobs.py \
    --mode fp32 \
    --output "$FP32_OUTPUT" \
    "${SCRIPT_ARGS[@]}"

# Step 3: Score with HuggingFace
echo ""
echo "============================================================"
echo "Step 3/4: HuggingFace scoring"
echo "============================================================"
uv run python scripts/analysis/fp32-lm-head/get_hf_logprobs.py \
    --bf16-input "$BF16_OUTPUT" \
    --fp32-input "$FP32_OUTPUT" \
    --output "$RESULTS_OUTPUT"

# Step 4: Plot results
echo ""
echo "============================================================"
echo "Step 4/4: Plotting"
echo "============================================================"
uv run python scripts/analysis/fp32-lm-head/plot_logprobs.py \
    --input "$RESULTS_OUTPUT" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "Done! Results saved to:"
echo "  vLLM BF16: $BF16_OUTPUT"
echo "  vLLM FP32: $FP32_OUTPUT"
echo "  Combined:  $RESULTS_OUTPUT"
echo "  Plots:     $OUTPUT_DIR/*.png"
echo "============================================================"
