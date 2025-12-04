#!/bin/bash

# Get the Beaker username to construct the image name
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/jupiter \
       --cluster ai2/augusta \
       --cluster ai2/saturn \
       --image "$BEAKER_IMAGE" \
       --description "GPU tests for test_grpo_fast_gpu.py" \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority high \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --budget ai2/oe-adapt \
       --no-host-networking \
       --gpus 1 \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& uv run pytest open_instruct/test_grpo_fast_gpu.py -xvs
