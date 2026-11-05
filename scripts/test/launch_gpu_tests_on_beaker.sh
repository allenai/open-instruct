#!/bin/bash
# Upload the locally built GPU test image to Beaker and launch the GPU tests.
# Prints the experiment ID and, when running in GitHub Actions, writes it to
# GITHUB_OUTPUT as `experiment_id`. Waiting happens separately, in
# wait_for_beaker_experiment.sh, so it can run on a small runner.
set -euo pipefail

GIT_SHA="${1:?Usage: $0 <git_sha>}"

beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')
short_sha="${GIT_SHA::8}"
image_name="open-instruct-gpu-tests-${short_sha}"

beaker image create open-instruct-gpu-tests:latest -n "$image_name" -w "ai2/$beaker_user" --description "Git commit: $GIT_SHA"

uv sync --frozen
output=$(bash scripts/test/run_gpu_pytest.sh "$beaker_user/$image_name" 2>&1)
echo "$output"

exp_id=$(echo "$output" | grep -oP 'https://beaker.org/ex/\K[^\s]+' | head -1)
if [ -z "$exp_id" ]; then
    echo "ERROR: could not find a Beaker experiment URL in the launch output"
    exit 1
fi
# For the workflow's cleanup step, in case the job is cancelled before its outputs are read.
if [ -n "${RUNNER_TEMP:-}" ]; then
    echo "$exp_id" > "$RUNNER_TEMP/gpu_test_experiment_id"
fi
echo ""
echo "Launched GPU tests: https://beaker.org/ex/${exp_id}"
if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "experiment_id=${exp_id}" >> "$GITHUB_OUTPUT"
fi
