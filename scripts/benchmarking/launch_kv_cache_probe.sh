#!/bin/bash
# Launches the vLLM KV-cache max-batch-size probe on a single 8-GPU node via mason.py.
# Usage (via build_image_and_launch.sh, which supplies the image name as $1):
#   ./scripts/train/build_image_and_launch.sh scripts/benchmarking/launch_kv_cache_probe.sh [model_name_or_path]
set -e

image_name="$1"
shift

model_name_or_path="${1:-Qwen/Qwen3.6-35B-A3B}"

git_hash=$(git rev-parse --short HEAD)
git_branch=$(git rev-parse --abbrev-ref HEAD)

uv run python mason.py \
    --cluster ai2/jupiter \
    --cluster ai2/ceres \
    --cluster ai2/saturn \
    --non_resumable \
    --image "$image_name" \
    --description "KV cache max batch size probe at commit $git_hash on branch $git_branch with model $model_name_or_path." \
    --pure_docker_mode \
    --workspace ai2/open-instruct-dev \
    --preemptible \
    --priority urgent \
    --num_nodes 1 \
    --max_retries 0 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env NCCL_CUMEM_ENABLE=0 \
    --gpus 8 \
    --secret HF_TOKEN=finbarrt_HF_TOKEN \
    --no_auto_dataset_cache \
    --task_name open_instruct-kv_cache_probe -- source configs/beaker_configs/ray_node_setup.sh \&\& python scripts/benchmarking/vllm_kv_cache_probe.py \
        --model_name_or_path "$model_name_or_path"
