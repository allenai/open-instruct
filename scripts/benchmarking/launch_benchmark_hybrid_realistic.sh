#!/bin/bash
# Runs the hybrid model benchmark with GRPO-realistic generation settings:
# - --vllm_enforce_eager (required for hybrid)
# - --stop_strings "</answer>" (variable-length responses, like GRPO)
# - 32 unique prompts x 16 samples (matching GRPO batch config)
#
# Usage:
#   ./scripts/train/build_image_and_launch.sh scripts/benchmarking/launch_benchmark_hybrid_realistic.sh
set -e

image_name="${1:-${BEAKER_USER}/open-instruct-integration-test}"
model_name_or_path="/weka/oe-training-default/ai2-llm/checkpoints/willm/linear-rnns/OLMo3.1-7B-6T-30h-long-context-drope/step23842-hf"

git_hash=$(git rev-parse --short HEAD)
git_branch=$(git rev-parse --abbrev-ref HEAD)

echo "Launching hybrid realistic benchmark at commit $git_hash on branch $git_branch"

uv run python mason.py \
    --cluster ai2/jupiter \
    --non_resumable \
    --image "$image_name" \
    --description "Hybrid benchmark with GRPO-realistic settings (enforce_eager, stop_strings, 32x16) at commit $git_hash on branch $git_branch." \
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
    --task_name open_instruct-benchmark_generators -- source configs/beaker_configs/ray_node_setup.sh \&\& python -m open_instruct.benchmark_generators \
        --model_name_or_path "$model_name_or_path" \
        --tokenizer_name_or_path "allenai/OLMo-2-1124-7B" \
        --dataset_mixer_list "saurabh5/rlvr_acecoder_filtered" "1.0" \
        --dataset_mixer_list_splits "train" \
        --max_prompt_token_length 2048 \
        --temperature 1.0 \
        --verbose True \
        --response_length 4096 \
        --stop_strings "</answer>" \
        --vllm_top_p 0.9 \
        --num_unique_prompts_rollout 32 \
        --num_samples_per_prompt_rollout 16 \
        --vllm_enforce_eager \
        --inflight_updates True \
        --vllm_num_engines 4 \
        --vllm_tensor_parallel_size 2 \
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

echo "Benchmark launched!"
