#!/bin/bash

BEAKER_IMAGE="${1:-finbarrt/open-instruct-integration-test}"

# Note: We do NOT source ray_node_setup.sh here because we want the test to
# start its own Ray cluster with the correct runtime_env (VLLM_ENABLE_V1_MULTIPROCESSING=0).
# If Ray is already running, the test's ray.init() is skipped.
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
    -- 'pytest open_instruct/test_grpo_fast_gpu.py::TestGeneration -xvs'
