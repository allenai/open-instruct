#!/bin/bash
# Run inference on all eval datasets with a tool-use model
#
# Usage:
#   ./scripts/run_all_eval_inference.sh <model_path> [output_dir]
#
# Example:
#   ./scripts/run_all_eval_inference.sh /path/to/checkpoint results/
#   ./scripts/run_all_eval_inference.sh allenai/Olmo-3-7B-Instruct-SFT
#
# Required environment variables:
#   SERPER_API_KEY - for web search
#   JINA_API_KEY   - for web browsing
#   S2_API_KEY     - for paper search
#
# To set from Beaker secrets:
#   BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
#   export SERPER_API_KEY=$(beaker secret read ${BEAKER_USER}_SERPER_API_KEY --workspace ai2/open-instruct-dev)
#   export JINA_API_KEY=$(beaker secret read ${BEAKER_USER}_JINA_API_KEY --workspace ai2/open-instruct-dev)
#   export S2_API_KEY=$(beaker secret read ${BEAKER_USER}_S2_API_KEY --workspace ai2/open-instruct-dev)

set -e

# Check required environment variables
if [ -z "$SERPER_API_KEY" ]; then
    echo "Error: SERPER_API_KEY environment variable is not set"
    exit 1
fi
if [ -z "$JINA_API_KEY" ]; then
    echo "Error: JINA_API_KEY environment variable is not set"
    exit 1
fi
if [ -z "$S2_API_KEY" ]; then
    echo "Error: S2_API_KEY environment variable is not set"
    exit 1
fi

# Export for child processes
export SERPER_API_KEY
export JINA_API_KEY
export S2_API_KEY

# Allow long context (matches training config)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

MODEL_PATH=${1:-"/weka/oe-adapt-default/allennlp/deletable_checkpoint/hamishivi/olmo3_7b_tool_use_test__1__1769424753_checkpoints/step_1500"}
OUTPUT_DIR=${2:-"results"}
DATA_DIR="data"

# Tool configuration (matches from_instruct_sft.sh)
TOOLS="python serper_search jina_browse s2_search"
TOOL_CALL_NAMES="code search browse paper_search"
TOOL_PARSER_TYPE="vllm_olmo3"
MAX_TOOL_CALLS=5
MAX_TOKENS=8192
TEMPERATURE=0.7

echo "=== Running Eval Inference ==="
echo "Model: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate eval data if not present
if [ ! -f "$DATA_DIR/aime.jsonl" ] || [ ! -f "$DATA_DIR/simpleqa.jsonl" ] || [ ! -f "$DATA_DIR/mbpp.jsonl" ] || [ ! -f "$DATA_DIR/gpqa.jsonl" ]; then
    echo "Generating eval data..."
    uv run python scripts/create_eval_data.py --all --output_dir "$DATA_DIR"
    echo ""
fi

# Run inference on each dataset
# Format: dataset_name:num_samples
DATASETS=("aime:32" "simpleqa:1" "mbpp:1" "gpqa:1")

for entry in "${DATASETS[@]}"; do
    dataset="${entry%%:*}"
    num_samples="${entry##*:}"
    
    INPUT_FILE="$DATA_DIR/${dataset}.jsonl"
    OUTPUT_FILE="$OUTPUT_DIR/${dataset}_results.jsonl"
    
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Warning: $INPUT_FILE not found, skipping..."
        continue
    fi
    
    echo "=== Processing $dataset (${num_samples} samples per prompt) ==="
    echo "Input: $INPUT_FILE"
    echo "Output: $OUTPUT_FILE"
    
    uv run python scripts/inference_with_tools.py \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE" \
        --model_name_or_path "$MODEL_PATH" \
        --tools $TOOLS \
        --tool_call_names $TOOL_CALL_NAMES \
        --tool_configs '{"api_endpoint": "https://open-instruct-tool-server-10554368204.us-central1.run.app/execute", "timeout": 30}' '{}' '{}' '{}' \
        --tool_parser_type "$TOOL_PARSER_TYPE" \
        --max_tool_calls "$MAX_TOOL_CALLS" \
        --max_tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --num_samples "$num_samples"
    
    echo "Completed $dataset"
    echo ""
done

echo "=== All Done ==="
echo "Results saved to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "No results found"
