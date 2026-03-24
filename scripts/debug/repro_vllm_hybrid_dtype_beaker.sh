#!/bin/bash
# Launch the vLLM hybrid dtype repro on Beaker with a single GPU.

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run --no-sync python mason.py \
    --cluster ai2/jupiter \
    --cluster ai2/saturn \
    --cluster ai2/ceres \
    --image "$BEAKER_IMAGE" \
    --description "Repro: vLLM hybrid dtype serialization bug" \
    --pure_docker_mode \
    --no-host-networking \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --num_nodes 1 \
    --max_retries 0 \
    --timeout 15m \
    --budget ai2/oe-adapt \
    --gpus 1 \
    --no_auto_dataset_cache \
    -- python scripts/debug/repro_vllm_hybrid_dtype.py
