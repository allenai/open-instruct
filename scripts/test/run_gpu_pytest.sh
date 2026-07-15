#!/bin/bash
set -eo pipefail

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
shift || true
PYTEST_ARGS=("$@")

echo "Using Beaker image: $BEAKER_IMAGE"
if [[ ${#PYTEST_ARGS[@]} -gt 0 ]]; then
    echo "Pytest filter: ${PYTEST_ARGS[*]}"
fi

# The CUDA 13 image requires NVIDIA driver >= 580, which currently only the
# B300 cluster (ai2/holmes) has. Re-add ai2/jupiter, ai2/ceres, and ai2/saturn
# once their drivers are upgraded.
uv run python mason.py \
       --cluster ai2/holmes \
       --image "$BEAKER_IMAGE" \
       --description "GPU tests for test_*_gpu.py" \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --no-host-networking \
       --gpus 1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)" \
       -- bash scripts/test/run_gpu_tests.sh "${PYTEST_ARGS[@]}"
