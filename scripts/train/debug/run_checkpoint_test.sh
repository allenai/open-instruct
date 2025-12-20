#!/bin/bash
# Launch the checkpoint restoration test on Beaker.
# This tests that num_total_tokens is correctly restored from checkpoint.

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --cluster ai2/saturn \
    --cluster ai2/ceres \
    --image "$BEAKER_IMAGE" \
    --description "Checkpoint restoration test" \
    --pure_docker_mode \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --num_nodes 1 \
    --max_retries 0 \
    --timeout 30m \
    --budget ai2/oe-adapt \
    --gpus 1 \
    -- bash scripts/train/debug/test_checkpoint_restoration.sh
