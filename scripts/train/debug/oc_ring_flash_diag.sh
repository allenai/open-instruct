#!/bin/bash
# One-GPU diagnostic job that verifies ring_flash_attn / flash_attn import
# inside the image produced by build_image_and_launch.sh.

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "ring_flash_attn / flash_attn import diagnostic" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --gpus 1 \
    --non_resumable \
    --no-host-networking \
    --no_auto_dataset_cache \
    -- \
    python scripts/train/debug/ring_flash_diag.py
