#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <image_name>"
    exit 1
fi

image_name="$1"
model_name_or_path="allenai/Olmo-Hybrid-Instruct-DPO-7B"
response_length=1024

git_hash=$(git rev-parse --short HEAD)
git_branch=$(git rev-parse --abbrev-ref HEAD)

for enforce_eager in "true" "false"; do
    eager_flag=""
    eager_desc="no-enforce-eager"
    if [ "$enforce_eager" = "true" ]; then
        eager_flag="--vllm_enforce_eager"
        eager_desc="enforce-eager"
    fi

    echo "Launching: response_length=$response_length, $eager_desc, tp=1"

    uv run python mason.py \
        --cluster ai2/jupiter \
        --non_resumable \
        --image "$image_name" \
        --description "Hybrid benchmark H100 TP1: response_length=$response_length, $eager_desc, commit=$git_hash, branch=$git_branch" \
        --pure_docker_mode \
        --workspace ai2/open-instruct-dev \
        --preemptible \
        --priority high \
        --num_nodes 1 \
        --max_retries 0 \
        --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        --env NCCL_CUMEM_ENABLE=0 \
        --budget ai2/oe-adapt \
        --gpus 8 \
        --secret HF_TOKEN=finbarrt_HF_TOKEN \
        --no_auto_dataset_cache \
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
            --vllm_num_engines 8 \
            --vllm_tensor_parallel_size 1 \
            --vllm_enable_prefix_caching \
            --vllm_gpu_memory_utilization 0.9 \
            --pack_length 40000 \
            --chat_template_name "tulu_thinker" \
            --trust_remote_code \
            --seed 42 \
            --add_bos \
            --dataset_local_cache_dir "benchmark_cache" \
            --dataset_cache_mode "local" \
            --dataset_transform_fn "rlvr_tokenize_v1" "rlvr_max_length_filter_v1" \
            $eager_flag

    echo "Launched: response_length=$response_length, $eager_desc"
    echo "----------------------------------------"
done

echo ""
echo "All H100 TP1 1k benchmarks launched successfully!"
