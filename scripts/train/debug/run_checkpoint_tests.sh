#!/bin/bash

BEAKER_IMAGE="$1"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/saturn \
       --image "$BEAKER_IMAGE" \
       --description "Checkpoint tests (2 GPUs, no pre-started Ray)" \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority normal \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --budget ai2/oe-adapt \
       --no-host-networking \
       --gpus 2 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)" \
       -- uv run pytest open_instruct/test_grpo_fast_gpu.py::TestCheckpointing -xvs \; uv run pytest open_instruct/test_grpo_fast_gpu.py::TestCheckpointingCPU -xvs
