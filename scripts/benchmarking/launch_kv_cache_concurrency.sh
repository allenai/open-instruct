#!/bin/bash
# Launches the KV cache concurrency script on Beaker for one or more models.
# Usage: ./scripts/train/build_image_and_launch.sh scripts/benchmarking/launch_kv_cache_concurrency.sh [model1] [model2] ...
set -e

image_name="$1"
shift

git_hash=$(git rev-parse --short HEAD)
git_branch=$(git rev-parse --abbrev-ref HEAD)

if [ "$#" -eq 0 ]; then
    set -- "/weka/oe-training-default/ai2-llm/checkpoints/willm/linear-rnns/OLMo3.1-7B-6T-30h-long-context-drope/step23842-hf"
fi

uv run python mason.py \
    --cluster ai2/jupiter \
    --non_resumable \
    --image "$image_name" \
    --description "KV cache concurrency analysis at commit $git_hash on branch $git_branch for $* models." \
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
    --task_name open_instruct-kv_cache_concurrency -- python scripts/benchmarking/kv_cache_concurrency.py \
        --model_name_or_path "$@" \
        --tensor_parallel_size 2 \
        --gpu_memory_utilization 0.9 \
        --trust_remote_code \
        --sequence_lengths 1024 4096 8192 16384 32768
