#!/bin/bash
set -e

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
model_name_or_path="allenai/Olmo-Hybrid-Instruct-DPO-7B"
response_length=4096

git_hash=$(git rev-parse --short HEAD)
git_branch=$(git rev-parse --abbrev-ref HEAD)

echo "Launching hybrid benchmark for model: $model_name_or_path"
echo "Response length: $response_length"
echo "----------------------------------------"

uv run python mason.py \
    --cluster ai2/jupiter \
    --non_resumable \
    --image "$BEAKER_IMAGE" \
    --description "Hybrid model benchmark, response_length=$response_length, commit $git_hash on $git_branch." \
    --pure_docker_mode \
    --workspace ai2/open-instruct-dev \
    --preemptible \
    --priority urgent \
    --num_nodes 1 \
    --max_retries 0 \
    --timeout 60m \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    --env NCCL_CUMEM_ENABLE=0 \
    --budget ai2/oe-adapt \
    --gpus 8 \
    --no_auto_dataset_cache \
    --secret HF_TOKEN=finbarrt_HF_TOKEN \
    --task_name open_instruct-benchmark_generators -- source configs/beaker_configs/ray_node_setup.sh \&\& python -m open_instruct.benchmark_generators \
        --model_name_or_path "$model_name_or_path" \
        --tokenizer_name_or_path "$model_name_or_path" \
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
        --vllm_num_engines 2 \
        --vllm_tensor_parallel_size 4 \
        --vllm_enable_prefix_caching \
        --vllm_gpu_memory_utilization 0.9 \
        --vllm_enforce_eager \
        --pack_length 40000 \
        --chat_template_name olmo123 \
        --trust_remote_code \
        --seed 42 \
        --add_bos \
        --dataset_local_cache_dir "benchmark_cache" \
        --dataset_cache_mode "local" \
        --dataset_transform_fn "rlvr_tokenize_v1" "rlvr_max_length_filter_v1"

echo "Hybrid benchmark launched successfully!"
