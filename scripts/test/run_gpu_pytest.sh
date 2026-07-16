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

case "${OPEN_INSTRUCT_CUDA_VERSION:-12}" in
    12) CLUSTER=ai2/jupiter ;;
    13) CLUSTER=ai2/holmes ;;
    *)
        echo "Error: OPEN_INSTRUCT_CUDA_VERSION must be 12 or 13."
        exit 1
        ;;
esac

echo "Using CUDA ${OPEN_INSTRUCT_CUDA_VERSION:-12} test cluster: $CLUSTER"
uv run python mason.py \
       --cluster "$CLUSTER" \
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
