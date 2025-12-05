#!/bin/bash

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/phobos-cirrascale \
    --cluster ai2/hammond-cirrascale \
    --image "$BEAKER_IMAGE" \
    --description "Test tool parsers (Hermes, OLMo3, get_triggered_tool)." \
    --pure_docker_mode \
    --workspace ai2/open-instruct-dev \
    --priority high \
    --preemptible \
    --num_nodes 1 \
    --max_retries 0 \
    --budget ai2/oe-adapt \
    --no-host-networking \
    --gpus 0 \
    -- pytest open_instruct/test_tool_parsers.py -v
