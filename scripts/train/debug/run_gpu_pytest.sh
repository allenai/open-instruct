#!/bin/bash

BEAKER_IMAGE="$1"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/saturn \
       --image "$BEAKER_IMAGE" \
       --description "GPU tests for test_grpo_fast_gpu.py" \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --budget ai2/oe-adapt \
       --no-host-networking \
       --gpus 1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)" \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& uv run pytest open_instruct/test_grpo_fast_gpu.py -xvs \; cp -r open_instruct/test_data /output/test_data 2\>/dev/null \|\| true
