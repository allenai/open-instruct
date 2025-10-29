#!/bin/bash
# Runs benchmarks for one or more models with a specified response length using mason.py.
# Can be called in two ways:
# 1. Directly: ./launch_benchmark.sh <image_name> <response_length> <model1> [model2] ...
# 2. Via build_image_and_launch.sh: build_image_and_launch.sh launch_benchmark.sh <response_length> <model1> [model2] ...
#    (in this case, build_image_and_launch.sh provides the image_name as the first argument)
set -e

# Check if at least 3 arguments are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <image_name> <response_length> <model1> [model2] ..."
    echo "Example: $0 nathanl/open_instruct_auto 64000 hamishivi/qwen2_5_openthoughts2 another/model"
    echo "Or via build_image_and_launch.sh: ./scripts/train/build_image_and_launch.sh scripts/launch_benchmark.sh 64 Qwen/Qwen2.5-7B"
    exit 1
fi

# First argument is the image name
image_name="$1"
shift

# Second argument is the response length
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

    uv run python mason.py \
        --cluster ai2/ceres \
        --cluster ai2/jupiter \
        --cluster ai2/saturn \
	      --non_resumable \
        --image "$image_name" \
        --description "Running single node benchmark with response length of $response_length at commit $git_hash on branch $git_branch with model $model_name_or_path." \
        --pure_docker_mode \
        --workspace ai2/open-instruct-dev \
      	--preemptible \
        --priority urgent \
        --num_nodes 1 \
        --max_retries 0 \
        --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        --env NCCL_CUMEM_ENABLE=0 \
        --budget ai2/oe-adapt \
        --gpus 8 \
        --secret HF_TOKEN=finbarrt_HF_TOKEN \
        --task_name open_instruct-benchmark_generators -- source configs/beaker_configs/ray_node_setup.sh \&\& python -m open_instruct.benchmark_generators \
            --model_name_or_path "$model_name_or_path" \
            --tokenizer_name_or_path "allenai/OLMo-2-1124-7B" \
            --dataset_mixer_list "hamishivi/hamishivi_rlvr_orz_math_57k_collected_all_filtered_hamishivi_qwen2_5_openthoughts2" "1.0" \
            --dataset_mixer_list_splits "train" \
            --max_prompt_token_length 2048 \
            --temperature 1.0 \
	          --verbose True \
            --response_length "$response_length" \
            --vllm_top_p 0.9 \
            --num_unique_prompts_rollout 16 \
            --num_samples_per_prompt_rollout 4 \
	    --inflight_updates True \
            --vllm_num_engines 1 \
            --vllm_tensor_parallel_size 4 \
            --vllm_enable_prefix_caching \
            --vllm_gpu_memory_utilization 0.9 \
            --pack_length 40000 \
            --chat_template_name "tulu_thinker" \
            --trust_remote_code \
            --seed 42 \
            --add_bos \
            --dataset_local_cache_dir "benchmark_cache" \
            --dataset_cache_mode "local" \
            --dataset_transform_fn "rlvr_tokenize_v1" "rlvr_max_length_filter_v1"

    echo "Launched benchmark for model: $model_name_or_path"
    echo "----------------------------------------"
done

echo ""
echo "All benchmarks launched successfully!"
