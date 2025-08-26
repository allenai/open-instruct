#!/bin/bash
# Runs the benchmark on gantry. Takes two arguments: response length and model.
# Usage: ./gantry_run_benchmark.sh [response_length] [model]
# E.g. $ ./gantry_run_benchmark.sh 64000 hamishivi/qwen2_5_openthoughts2
set -e

# Set default values
response_length=64000
model_name_or_path="hamishivi/qwen2_5_openthoughts2"

# If first argument exists and is a number, use it as response_length
if [[ "$1" =~ ^[0-9]+$ ]]; then
  response_length="$1"
  shift
fi

# If second argument exists, use it as model
if [[ -n "$1" ]]; then
  model_name_or_path="$1"
  shift
fi

git_hash=$(git rev-parse --short HEAD)
git_branch=$(git rev-parse --abbrev-ref HEAD) \

gantry run \
       --name open_instruct-benchmark_generators \
       --workspace ai2/open-instruct-dev \
       --gpus 1 \
       --description "Running benchmark with response length of $response_length at commit $git_hash on branch $git_branch with model $model_name_or_path." \
       --beaker-image nathanl/open_instruct_auto \
       --priority urgent \
       --cluster ai2/augusta \
       --cluster ai2/jupiter \
       --cluster ai2/ceres \
       --budget ai2/oe-adapt \
       --install 'uv sync' \
       -- uv run python -m open_instruct.benchmark_generators \
    --model_name_or_path "$model_name_or_path" \
    --tokenizer_name_or_path "$model_name_or_path" \
    --dataset_mixer_list "hamishivi/hamishivi_rlvr_orz_math_57k_collected_all_filtered_hamishivi_qwen2_5_openthoughts2" "1.0" \
    --dataset_mixer_list_splits "train" \
    --max_token_length 10240 \
    --max_prompt_token_length 2048 \
    --temperature 1.0 \
    --response_length "$response_length" \
    --vllm_top_p 0.9 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 16 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.9 \
    --pack_length 40000 \
    --chat_template_name "tulu_thinker" \
    --trust_remote_code \
    --seed 42 \
    --dataset_local_cache_dir "benchmark_cache" \
    --dataset_cache_mode "local" \
    --dataset_transform_fn "rlvr_tokenize_v1" "rlvr_filter_v1"
