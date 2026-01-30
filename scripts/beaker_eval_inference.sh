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
MODEL_PATH=${2:-"allenai/OLMo-3-8B-Instruct"}
OUTPUT_DIR=${3:-"/output/results"}
DATA_DIR="/output/data"

echo "Using image: $IMAGE"
echo "Model path: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR"

uv run python mason.py \
    --cluster ai2/ceres \
    --image "$IMAGE" \
    --description "Tool use eval inference" \
    --pure_docker_mode \
    --workspace ai2/tulu-thinker \
    --priority urgent \
    --num_nodes 1 \
    --max_retries 0 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
    --budget ai2/oe-adapt \
    --secret SERPER_API_KEY=${BEAKER_USER}_SERPER_API_KEY \
    --secret JINA_API_KEY=${BEAKER_USER}_JINA_API_KEY \
    --secret S2_API_KEY=${BEAKER_USER}_S2_API_KEY \
    --gpus 1 \
    -- mkdir -p ${OUTPUT_DIR} ${DATA_DIR} \&\& \
       python scripts/create_eval_data.py --all --output_dir ${DATA_DIR} \&\& \
       python scripts/inference_with_tools.py \
           --input_file ${DATA_DIR}/aime.jsonl \
           --output_file ${OUTPUT_DIR}/aime_results.jsonl \
           --model_name_or_path ${MODEL_PATH} \
           --tools python serper_search jina_browse s2_search \
           --tool_call_names code search browse paper_search \
           --tool_configs '{"api_endpoint": "https://open-instruct-tool-server-10554368204.us-central1.run.app/execute", "timeout": 30}' '{}' '{}' '{}' \
           --tool_parser_type vllm_olmo3 \
           --max_tool_calls 5 \
           --max_tokens 8192 \
           --temperature 0.7 \
           --num_samples 32 \&\& \
       python scripts/inference_with_tools.py \
           --input_file ${DATA_DIR}/simpleqa.jsonl \
           --output_file ${OUTPUT_DIR}/simpleqa_results.jsonl \
           --model_name_or_path ${MODEL_PATH} \
           --tools python serper_search jina_browse s2_search \
           --tool_call_names code search browse paper_search \
           --tool_configs '{"api_endpoint": "https://open-instruct-tool-server-10554368204.us-central1.run.app/execute", "timeout": 30}' '{}' '{}' '{}' \
           --tool_parser_type vllm_olmo3 \
           --max_tool_calls 5 \
           --max_tokens 8192 \
           --temperature 0.7 \
           --num_samples 1 \&\& \
       python scripts/inference_with_tools.py \
           --input_file ${DATA_DIR}/mbpp.jsonl \
           --output_file ${OUTPUT_DIR}/mbpp_results.jsonl \
           --model_name_or_path ${MODEL_PATH} \
           --tools python serper_search jina_browse s2_search \
           --tool_call_names code search browse paper_search \
           --tool_configs '{"api_endpoint": "https://open-instruct-tool-server-10554368204.us-central1.run.app/execute", "timeout": 30}' '{}' '{}' '{}' \
           --tool_parser_type vllm_olmo3 \
           --max_tool_calls 5 \
           --max_tokens 8192 \
           --temperature 0.7 \
           --num_samples 1 \&\& \
       python scripts/inference_with_tools.py \
           --input_file ${DATA_DIR}/gpqa.jsonl \
           --output_file ${OUTPUT_DIR}/gpqa_results.jsonl \
           --model_name_or_path ${MODEL_PATH} \
           --tools python serper_search jina_browse s2_search \
           --tool_call_names code search browse paper_search \
           --tool_configs '{"api_endpoint": "https://open-instruct-tool-server-10554368204.us-central1.run.app/execute", "timeout": 30}' '{}' '{}' '{}' \
           --tool_parser_type vllm_olmo3 \
           --max_tool_calls 5 \
           --max_tokens 8192 \
           --temperature 0.7 \
           --num_samples 1
