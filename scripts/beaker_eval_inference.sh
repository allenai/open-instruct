#!/bin/bash
# Run inference on all eval datasets via Beaker
#
# Usage (via build_image_and_launch.sh):
#   ./scripts/train/build_image_and_launch.sh scripts/beaker_eval_inference.sh [model_path] [output_dir]
#
# Example:
#   ./scripts/train/build_image_and_launch.sh scripts/beaker_eval_inference.sh
#   ./scripts/train/build_image_and_launch.sh scripts/beaker_eval_inference.sh /weka/path/to/checkpoint /output/results

set -e

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')

# First arg is the image (passed by build_image_and_launch.sh)
IMAGE=${1:-"hamishivi/open_instruct_dev_2302"}
MODEL_PATH=${2:-"/weka/oe-adapt-default/allennlp/deletable_checkpoint/hamishivi/olmo3_7b_tool_use_test__1__1769424753_checkpoints/step_1500"}
OUTPUT_DIR=${3:-"/output/results"}
DATA_DIR="/output/data"

echo "Using image: $IMAGE"
echo "Model path: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR"

uv run python mason.py \
    --cluster ai2/jupiter \
    --image "$IMAGE" \
    --description "Tool use eval inference" \
    --pure_docker_mode \
    --workspace ai2/tulu-thinker \
    --priority normal \
    --preemptible \
    --num_nodes 1 \
    --max_retries 0 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
    --budget ai2/oe-adapt \
    --secret SERPER_API_KEY=${BEAKER_USER}_SERPER_API_KEY \
    --secret JINA_API_KEY=${BEAKER_USER}_JINA_API_KEY \
    --secret S2_API_KEY=${BEAKER_USER}_S2_API_KEY \
    --gpus 1 \
    -- bash -c "
set -e

# Tool configuration
TOOLS='python serper_search jina_browse s2_search'
TOOL_CALL_NAMES='code search browse paper_search'
TOOL_PARSER_TYPE='vllm_olmo3'
MAX_TOOL_CALLS=5
MAX_TOKENS=8192
TEMPERATURE=0.7

MODEL_PATH='${MODEL_PATH}'
OUTPUT_DIR='${OUTPUT_DIR}'
DATA_DIR='${DATA_DIR}'

echo '=== Running Eval Inference ==='
echo \"Model: \$MODEL_PATH\"
echo \"Output directory: \$OUTPUT_DIR\"
echo ''

# Create directories
mkdir -p \"\$OUTPUT_DIR\"
mkdir -p \"\$DATA_DIR\"

# Generate eval data
echo 'Generating eval data...'
python scripts/create_eval_data.py --all --output_dir \"\$DATA_DIR\"
echo ''

# Run inference on each dataset
# Format: dataset_name:num_samples
declare -a DATASETS=('aime:32' 'simpleqa:1' 'mbpp:1' 'gpqa:1')

for entry in \"\${DATASETS[@]}\"; do
    dataset=\"\${entry%%:*}\"
    num_samples=\"\${entry##*:}\"
    
    INPUT_FILE=\"\$DATA_DIR/\${dataset}.jsonl\"
    OUTPUT_FILE=\"\$OUTPUT_DIR/\${dataset}_results.jsonl\"
    
    if [ ! -f \"\$INPUT_FILE\" ]; then
        echo \"Warning: \$INPUT_FILE not found, skipping...\"
        continue
    fi
    
    echo \"=== Processing \$dataset (\${num_samples} samples per prompt) ===\"
    echo \"Input: \$INPUT_FILE\"
    echo \"Output: \$OUTPUT_FILE\"
    
    python scripts/inference_with_tools.py \\
        --input_file \"\$INPUT_FILE\" \\
        --output_file \"\$OUTPUT_FILE\" \\
        --model_name_or_path \"\$MODEL_PATH\" \\
        --tools \$TOOLS \\
        --tool_call_names \$TOOL_CALL_NAMES \\
        --tool_configs '{\"api_endpoint\": \"https://open-instruct-tool-server-10554368204.us-central1.run.app/execute\", \"timeout\": 30}' '{}' '{}' '{}' \\
        --tool_parser_type \"\$TOOL_PARSER_TYPE\" \\
        --max_tool_calls \"\$MAX_TOOL_CALLS\" \\
        --max_tokens \"\$MAX_TOKENS\" \\
        --temperature \"\$TEMPERATURE\" \\
        --num_samples \"\$num_samples\"
    
    echo \"Completed \$dataset\"
    echo ''
done

echo '=== All Done ==='
echo \"Results saved to: \$OUTPUT_DIR\"
ls -la \"\$OUTPUT_DIR\"/*.jsonl 2>/dev/null || echo 'No results found'
"
