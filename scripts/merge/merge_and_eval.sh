#!/bin/bash
#
# Automated model merging and evaluation script
#
# Usage:
#   ./scripts/merge/merge_and_eval.sh [config_file] [output_dir] [model_name]
#
# Example:
#   ./scripts/merge/merge_and_eval.sh configs/my_merge.yaml /weka/oe-adapt-default/nathanl/merged/my_merge my-merged-model
#
# Or run with defaults to merge the best SFT checkpoints:
#   ./scripts/merge/merge_and_eval.sh
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Default paths (override with command line args)
BEAKER_USER=$(beaker account whoami --format json 2>/dev/null | jq -r '.[0].name' || echo "nathanl")
BASE_CHECKPOINT_PATH="/weka/oe-adapt-default/${BEAKER_USER}/checkpoints"
DEFAULT_OUTPUT_DIR="/weka/oe-adapt-default/${BEAKER_USER}/merged"

# Parse arguments
CONFIG_FILE="${1:-}"
OUTPUT_DIR="${2:-${DEFAULT_OUTPUT_DIR}/merged-$(date +%Y%m%d-%H%M%S)}"
MODEL_NAME="${3:-merged-model-$(date +%Y%m%d-%H%M%S)}"

# ============================================================================
# Step 1: Create merge config if not provided
# ============================================================================

if [ -z "$CONFIG_FILE" ]; then
    echo "No config file provided. Creating default merge config..."

    # Create a temporary config for merging the best checkpoints
    CONFIG_FILE=$(mktemp /tmp/merge_config_XXXXXX.yaml)

    # Default: merge the last 2-3 best checkpoints
    # Edit these paths to match your actual checkpoint locations
    cat > "$CONFIG_FILE" << 'EOF'
# Auto-generated merge config
# Edit the model paths below to match your checkpoints

models:
  # Add your model paths here
  # Example:
  # - model: /weka/oe-adapt-default/nathanl/checkpoints/MODEL_A/step46412-hf
  #   parameters:
  #     weight: 1.0
  # - model: /weka/oe-adapt-default/nathanl/checkpoints/MODEL_B/step46412-hf
  #   parameters:
  #     weight: 1.0

merge_method: linear
dtype: bfloat16
EOF

    echo "Created config at: $CONFIG_FILE"
    echo "Please edit the config file to add your model paths, then re-run this script."
    echo ""
    echo "Example usage after editing:"
    echo "  $0 $CONFIG_FILE $OUTPUT_DIR $MODEL_NAME"
    exit 1
fi

echo "=============================================="
echo "Model Merging and Evaluation"
echo "=============================================="
echo "Config:     $CONFIG_FILE"
echo "Output:     $OUTPUT_DIR"
echo "Model name: $MODEL_NAME"
echo "=============================================="

# ============================================================================
# Step 2: Verify mergekit is installed
# ============================================================================

if ! command -v mergekit-yaml &> /dev/null; then
    echo "ERROR: mergekit-yaml not found!"
    echo ""
    echo "Install mergekit first:"
    echo "  git clone https://github.com/arcee-ai/mergekit.git"
    echo "  cd mergekit && pip install -e ."
    exit 1
fi

# ============================================================================
# Step 3: Run the merge
# ============================================================================

echo ""
echo "Running merge..."
echo ""

mergekit-yaml "$CONFIG_FILE" "$OUTPUT_DIR"

echo ""
echo "Merge complete! Output at: $OUTPUT_DIR"

# ============================================================================
# Step 4: Verify and copy tokenizer if needed
# ============================================================================

echo ""
echo "Checking tokenizer..."

if [ ! -f "$OUTPUT_DIR/tokenizer.json" ] && [ ! -f "$OUTPUT_DIR/tokenizer.model" ]; then
    echo "WARNING: Tokenizer not found in output directory!"

    # Try to extract source model path from config and copy tokenizer
    SOURCE_MODEL=$(grep -A1 "model:" "$CONFIG_FILE" | grep -v "model:" | head -1 | tr -d ' ' | tr -d '-')

    if [ -n "$SOURCE_MODEL" ] && [ -d "$SOURCE_MODEL" ]; then
        echo "Copying tokenizer from: $SOURCE_MODEL"
        cp -v "$SOURCE_MODEL"/tokenizer* "$OUTPUT_DIR/" 2>/dev/null || true
        cp -v "$SOURCE_MODEL"/special_tokens_map.json "$OUTPUT_DIR/" 2>/dev/null || true
        cp -v "$SOURCE_MODEL"/tokenizer_config.json "$OUTPUT_DIR/" 2>/dev/null || true
    else
        echo "Could not auto-copy tokenizer. Please copy manually from a source model."
    fi
else
    echo "Tokenizer found in output directory."
fi

# ============================================================================
# Step 5: Run evaluations
# ============================================================================

echo ""
echo "=============================================="
echo "Submitting evaluation jobs..."
echo "=============================================="

# Batch 1: Standard evals
uv run scripts/submit_eval_jobs.py \
    --model_name "${MODEL_NAME}" \
    --location "${OUTPUT_DIR}" \
    --cluster ai2/jupiter ai2/ceres \
    --is_tuned \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --use_hf_tokenizer_template \
    --beaker_image yanhongl/oe_eval_olmo3_devel_v5 \
    --oe_eval_tasks "gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,omega_500:0-shot-chat_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,ifeval_ood::tulu-thinker-deepseek" \
    --run_oe_eval_experiments \
    --evaluate_on_weka \
    --run_id placeholder \
    --oe_eval_max_length 32768 \
    --process_output r1_style \
    --skip_oi_evals

# Batch 2: BBH, MMLU, etc (requires more GPU memory)
uv run scripts/submit_eval_jobs.py \
    --model_name "${MODEL_NAME}" \
    --location "${OUTPUT_DIR}" \
    --cluster ai2/jupiter ai2/ceres \
    --is_tuned \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --gpu_multiplier 2 \
    --preemptible \
    --use_hf_tokenizer_template \
    --beaker_image yanhongl/oe_eval_olmo3_devel_v5 \
    --oe_eval_tasks "bbh:cot::hamish_zs_reasoning_deepseek_v2,mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek" \
    --run_oe_eval_experiments \
    --evaluate_on_weka \
    --run_id placeholder \
    --oe_eval_max_length 32768 \
    --process_output r1_style \
    --skip_oi_evals

echo ""
echo "=============================================="
echo "All done!"
echo "=============================================="
echo "Merged model: $OUTPUT_DIR"
echo "Eval jobs submitted for: $MODEL_NAME"
echo ""
