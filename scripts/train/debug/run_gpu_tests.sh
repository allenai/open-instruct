#!/bin/bash

BEAKER_IMAGE="${1:-finbarrt/open-instruct-integration-test}"

uv run python mason.py \
    --cluster ai2/saturn \
    --image "$BEAKER_IMAGE" \
    --description "GPU test: test_grpo_fast_gpu.py" \
    --pure_docker_mode \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --num_nodes 1 \
    --max_retries 0 \
    --timeout 30m \
    --budget ai2/oe-adapt \
    --gpus 1 \
    -- 'source configs/beaker_configs/ray_node_setup.sh && export VLLM_ENABLE_V1_MULTIPROCESSING=0 && export VLLM_LOGGING_LEVEL=DEBUG && echo "=== /dev/shm size ===" && df -h /dev/shm && echo "=====================" && pytest open_instruct/test_grpo_fast_gpu.py::TestGeneration::test_generation_deterministic_1 -xvs'
