#!/bin/bash
# Runs benchmarks for one or more models with a specified response length on gantry.
# Usage: ./gantry_run_benchmark.sh <response_length> <model1> [model2] ...
# E.g. $ ./gantry_run_benchmark.sh 64000 hamishivi/qwen2_5_openthoughts2 another/model
set -e

# Check if at least 2 arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <response_length> <model1> [model2] ..."
    echo "Example: $0 64000 hamishivi/qwen2_5_openthoughts2 another/model"
    exit 1
fi

# First argument is the response length
response_length="$1"
shift

# Validate that response_length is a number
if ! [[ "$response_length" =~ ^[0-9]+$ ]]; then
    echo "Error: Response length must be a positive integer"
    exit 1
fi

echo "Running benchmarks with response length: $response_length"
echo "Models to benchmark: $@"
echo "----------------------------------------"

git_hash=$(git rev-parse --short HEAD)
git_branch=$(git rev-parse --abbrev-ref HEAD)

# Loop through remaining arguments (models)
for model_name_or_path in "$@"; do
    echo ""
    echo "Launching benchmark for model: $model_name_or_path"

    gantry run \
           --name open_instruct-benchmark_generators \
           --workspace ai2/tulu-thinker \
           --gpus 1 \
           --description "Running benchmark with response length of $response_length at commit $git_hash on branch $git_branch with model $model_name_or_path." \
	   --beaker-image nathanl/open_instruct_auto \
	   --cluster ai2/prior-elanding \
           --budget ai2/oe-adapt \
           --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
           --env-secret="HF_TOKEN=finbarrt_HF_TOKEN" \
           -- uv run python -m open_instruct.benchmark_generators \
        --model_name_or_path "$model_name_or_path" \
        --tokenizer_name_or_path "allenai/OLMo-2-1124-7B" \
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
        --add_bos \
        --dataset_local_cache_dir "benchmark_cache" \
        --dataset_cache_mode "local" \
        --dataset_transform_fn "rlvr_tokenize_v1" "rlvr_filter_v1"

    echo "Launched benchmark for model: $model_name_or_path"
    echo "----------------------------------------"
done

echo ""
echo "All benchmarks launched successfully!"
