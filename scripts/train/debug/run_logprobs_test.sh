#!/bin/bash

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/jupiter \
       --cluster ai2/saturn \
       --cluster ai2/ceres \
       --image "$BEAKER_IMAGE" \
       --description "Logprobs comparison test (hybrid model)" \
       --pure_docker_mode \
       --no-host-networking \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --num_nodes 1 \
       --max_retries 0 \
       --timeout 30m \
       --budget ai2/oe-adapt \
       --gpus 1 \
       --no_auto_dataset_cache \
       -- uv run pytest tests/test_logprobs_comparison.py -v -x
