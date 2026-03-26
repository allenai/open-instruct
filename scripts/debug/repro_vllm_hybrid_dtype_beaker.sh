#!/bin/bash
set -e
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

uv run python mason.py \
    --cluster ai2/jupiter \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 1 \
    --timeout 30m \
    --max_retries 0 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --description "Repro: vLLM 0.18.0 hybrid dtype serialization bug" \
    --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    --gpus 2 -- python scripts/debug/repro_vllm_hybrid_dtype.py
