#!/bin/bash
# Run inference with a tool-use model trained with from_instruct_sft.sh
#
# Usage:
#   ./scripts/run_tool_inference.sh <model_path> <input_file> <output_file> [num_samples]
#
# Example:
#   ./scripts/run_tool_inference.sh /path/to/checkpoint data/aime.jsonl results/aime_results.jsonl
#
# With multiple samples per prompt (for majority voting):
#   ./scripts/run_tool_inference.sh /path/to/checkpoint data/aime.jsonl results/aime_results.jsonl 32
#
# Or with defaults (uses Olmo-3-7B-Instruct-SFT):
#   ./scripts/run_tool_inference.sh "" data/aime.jsonl results/aime_results.jsonl
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

MODEL_PATH=${1:-"allenai/Olmo-3-7B-Instruct-SFT"}
INPUT_FILE=${2:-"data/aime.jsonl"}
OUTPUT_FILE=${3:-"results/output.jsonl"}
NUM_SAMPLES=${4:-1}

# Tool configuration (matches from_instruct_sft.sh)
TOOLS="python serper_search jina_browse s2_search"
TOOL_CALL_NAMES="code search browse paper_search"
TOOL_PARSER_TYPE="vllm_olmo3"
MAX_TOOL_CALLS=5

# Generation settings
MAX_TOKENS=8192
TEMPERATURE=0.7

echo "=== Tool Inference ==="
echo "Model: $MODEL_PATH"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Tools: $TOOLS"
echo "Num samples: $NUM_SAMPLES"
echo ""

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Run inference
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
    --num_samples "$NUM_SAMPLES"

echo ""
echo "Done! Results saved to: $OUTPUT_FILE"
