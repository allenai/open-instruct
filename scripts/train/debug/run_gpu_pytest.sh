#!/bin/bash
set -xeo pipefail

echo "=== Debug info ==="
echo "PWD: $(pwd)"
echo "BEAKER_TOKEN set: ${BEAKER_TOKEN:+yes}"
echo "Args: $@"
echo "=================="

echo "Getting beaker user..."
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
echo "BEAKER_USER: $BEAKER_USER"

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
echo "Using Beaker image: $BEAKER_IMAGE"

echo "Starting mason.py..."
uv run python mason.py \
       --cluster ai2/jupiter \
       --cluster ai2/ceres \
       --cluster ai2/saturn \
       --image "$BEAKER_IMAGE" \
       --description "GPU tests for test_*_gpu.py" \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority high \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --budget ai2/oe-adapt \
       --no-host-networking \
       --gpus 1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)" \
       -- bash scripts/train/debug/run_gpu_tests.sh
