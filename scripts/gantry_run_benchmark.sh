#!/bin/bash
# Runs the benchmark on gantry with configurable response length and model.
#
# Usage: ./gantry_run_benchmark.sh [response_length] [model_name]
#
# Arguments:
#   response_length (optional): Maximum response length in tokens (default: 64000)
#   model_name (optional): Model name or path to use (default: hamishivi/qwen2_5_openthoughts2)
#
# Examples:
#   ./gantry_run_benchmark.sh                    # Use defaults
#   ./gantry_run_benchmark.sh 32000              # Custom response length
#   ./gantry_run_benchmark.sh 32000 my_model     # Custom response length and model
#
set -euo pipefail

# Set default values
response_length=64000
model_name="hamishivi/qwen2_5_openthoughts2"

# Parse first argument if it's a number (response_length)
if [[ $# -ge 1 ]] && [[ "$1" =~ ^[0-9]+$ ]]; then
  response_length="$1"
  shift
fi

# Parse second argument if provided (model_name)
if [[ $# -ge 1 ]]; then
  model_name="$1"
  shift
fi

# Calculate pack_length dynamically
max_prompt_token_length=2048
pack_length=$((max_prompt_token_length + response_length))

gantry run \
       --name open_instruct-benchmark_generators \
       --workspace ai2/oe-eval \
       --weka=oe-eval-default:/weka \
       --gpus 1 \
       --beaker-image nathanl/open_instruct_auto \
       --cluster ai2/jupiter-cirrascale-2 \
       --budget ai2/oe-eval \
       --install 'uv sync && uv run python -m nltk.downloader punkt' \
       -- uv run python -m open_instruct.benchmark_generators \
    --model_name_or_path "$model_name" \
    --tokenizer_name_or_path "$model_name" \
    --dataset_mixer_list "TTTXXX01/MathSub-30K" "1.0" \
    --dataset_mixer_list_splits "train" \
    --max_token_length 10240 \
    --max_prompt_token_length "$max_prompt_token_length" \
    --temperature 1.0 \
    --response_length "$response_length" \
    --vllm_top_p 0.9 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.9 \
    --pack_length "$pack_length" \
    --chat_template_name "tulu_thinker" \
    --trust_remote_code \
    --seed 42 \
    --dataset_local_cache_dir "benchmark_cache" \
    --dataset_cache_mode "local" \
    --dataset_transform_fn "rlvr_tokenize_v1" "rlvr_filter_v1" \
    --add_bos
