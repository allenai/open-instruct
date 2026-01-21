#!/bin/bash
set -euo pipefail

GIT_SHA="${1:?Usage: $0 <git_sha>}"

beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')
short_sha="${GIT_SHA::8}"
image_name="open-instruct-gpu-tests-${short_sha}"

beaker image create open-instruct-gpu-tests:latest -n "$image_name" -w "ai2/$beaker_user" --description "Git commit: $GIT_SHA"

uv sync --frozen
output=$(bash scripts/test/run_gpu_pytest.sh "$beaker_user/$image_name" 2>&1)
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
