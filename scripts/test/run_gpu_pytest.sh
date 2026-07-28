#!/bin/bash
set -eo pipefail

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
shift || true
PYTEST_ARGS=("$@")
GPU_COUNT="${GPU_COUNT:-1}"

echo "Using Beaker image: $BEAKER_IMAGE"
if [[ ${#PYTEST_ARGS[@]} -gt 0 ]]; then
    echo "Pytest filter: ${PYTEST_ARGS[*]}"
fi

case "$BEAKER_IMAGE" in
    *-cuda13)
        CUDA_VERSION=13
        CLUSTERS=(ai2/holmes)
        ;;
    *)
        CUDA_VERSION=12
        CLUSTERS=(ai2/jupiter ai2/ceres ai2/saturn)
        ;;
esac

CLUSTER_ARGS=()
for cluster in "${CLUSTERS[@]}"; do
    CLUSTER_ARGS+=(--cluster "$cluster")
done

echo "Using CUDA $CUDA_VERSION test clusters: ${CLUSTERS[*]}"
uv run python mason.py \
       "${CLUSTER_ARGS[@]}" \
       --image "$BEAKER_IMAGE" \
       --description "CUDA $CUDA_VERSION GPU tests for test_*_gpu.py" \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --no-host-networking \
       --gpus "$GPU_COUNT" \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)" \
       -- bash scripts/test/run_gpu_tests.sh "${PYTEST_ARGS[@]}"
