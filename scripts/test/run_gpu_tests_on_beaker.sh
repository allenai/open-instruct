#!/bin/bash
set -euo pipefail

GIT_SHA="${1:?Usage: $0 <git_sha> [cuda_version]}"
CUDA_VERSION="${2:-12}"
if [[ "$CUDA_VERSION" != "12" && "$CUDA_VERSION" != "13" ]]; then
    echo "Error: CUDA version must be 12 or 13."
    exit 1
fi

beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')
short_sha="${GIT_SHA::8}"
image_name="open-instruct-gpu-tests-${short_sha}-cuda${CUDA_VERSION}"

beaker image create open-instruct-gpu-tests:latest -n "$image_name" -w "ai2/$beaker_user" \
    --description "Git commit: $GIT_SHA; CUDA: $CUDA_VERSION"

uv sync --frozen
output=$(OPEN_INSTRUCT_CUDA_VERSION="$CUDA_VERSION" bash scripts/test/run_gpu_pytest.sh "$beaker_user/$image_name" 2>&1)
echo "$output"

exp_url=$(echo "$output" | grep -oP 'https://beaker.org/ex/[^\s]+' | head -1)
exp_id=$(echo "$exp_url" | grep -oP 'https://beaker.org/ex/\K[^\s]+')
echo ""
echo "=========================================="
echo "Waiting for tests to finish on Beaker: $exp_url"
echo "=========================================="
echo ""

beaker experiment await "$exp_id" 0 --index finalized --timeout 25m
status=$(beaker experiment get "$exp_id" --format json | jq -r '.[0].jobs[0].status.exitCode')
if [ "$status" != "0" ]; then
    echo "GPU tests failed with exit code $status"
    exit 1
fi
