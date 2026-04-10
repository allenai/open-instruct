#!/bin/bash

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/jupiter \
       --cluster ai2/saturn \
       --image "$BEAKER_IMAGE" \
       --description "Minimal weight sync NaN repro (AsyncLLMEngine + TP=2 + packed=False)" \
       --pure_docker_mode \
       --no-host-networking \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --num_nodes 1 \
       --max_retries 0 \
       --timeout 15m \
       --budget ai2/oe-adapt \
       --gpus 3 \
       --no_auto_dataset_cache \
       -- python scripts/debug/minimal_weight_sync_repro.py
