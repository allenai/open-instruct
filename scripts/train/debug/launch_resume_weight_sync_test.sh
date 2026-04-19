#!/bin/bash
# Launches the two-phase resume weight sync test on Beaker.
# Uses 3 GPUs: 2 learners (ZeRO-3, real cross-GPU parameter sharding) + 1 vLLM.
# This tests that GatheredParameters works on checkpoint-loaded ZeRO-3 weights
# without a prior training step.

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --cluster ai2/saturn \
    --image "$BEAKER_IMAGE" \
    --description "Resume weight sync test: ZeRO-3 2-learner eager checkpoint sync to vLLM." \
    --pure_docker_mode \
    --no-host-networking \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 1 \
    --max_retries 0 \
    --timeout 45m \
    --budget ai2/oe-adapt \
    --gpus 3 \
    --no_auto_dataset_cache \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& bash scripts/train/debug/beaker_test_resume_weight_sync.sh
