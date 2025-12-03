#!/bin/bash

BEAKER_IMAGE="${1:-finbarrt/open-instruct-integration-test}"

# Source ray_node_setup.sh to start Ray externally (like production), then run test.
# This avoids issues with vLLM V1 multiprocessing spawn inside inline Ray.
uv run python mason.py \
    --cluster ai2/saturn \
    --image "$BEAKER_IMAGE" \
    --description "GPU test: test_grpo_fast_gpu.py generation tests" \
    --pure_docker_mode \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --num_nodes 1 \
    --max_retries 0 \
    --timeout 30m \
    --budget ai2/oe-adapt \
    --gpus 1 \
    -- 'source configs/beaker_configs/ray_node_setup.sh && pytest open_instruct/test_grpo_fast_gpu.py::TestGeneration -xvs'
