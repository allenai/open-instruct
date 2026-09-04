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

# The image is CUDA 13.0 only, so it needs a cluster whose driver supports it.
# Add ai2/jupiter, ai2/ceres and ai2/saturn back here once their driver rollout
# lands, or override with GPU_TEST_CLUSTERS="ai2/foo ai2/bar".
read -r -a CLUSTERS <<< "${GPU_TEST_CLUSTERS:-ai2/holmes}"

CLUSTER_ARGS=()
for cluster in "${CLUSTERS[@]}"; do
    CLUSTER_ARGS+=(--cluster "$cluster")
done

echo "Using GPU test clusters: ${CLUSTERS[*]}"
uv run python mason.py \
       "${CLUSTER_ARGS[@]}" \
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
