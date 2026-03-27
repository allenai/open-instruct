#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <image_name>"
    exit 1
fi

image_name="$1"
git_hash=$(git rev-parse --short HEAD)
git_branch=$(git rev-parse --abbrev-ref HEAD)

uv run python mason.py \
    --cluster ai2/jupiter \
    --non_resumable \
    --image "$image_name" \
    --description "KV cache concurrency analysis at commit $git_hash on branch $git_branch for allenai/Olmo-Hybrid-7B models." \
    --pure_docker_mode \
    --workspace ai2/open-instruct-dev \
    --preemptible \
    --priority high \
    --num_nodes 1 \
    --max_retries 0 \
    --budget ai2/oe-adapt \
    --gpus 8 \
    --secret HF_TOKEN=finbarrt_HF_TOKEN \
    --no_auto_dataset_cache \
    --env VLLM_USE_V1=1 \
    --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    --env VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    --env NCCL_CUMEM_ENABLE=0 \
    --task_name open_instruct-kv_cache_concurrency \
    -- python scripts/benchmarking/kv_cache_concurrency.py \
        --model_name_or_path allenai/Olmo-Hybrid-7B \
        --tensor_parallel_size 2 \
        --gpu_memory_utilization 0.9 \
        --sequence_lengths 1024 4096 8192 16384 32768
