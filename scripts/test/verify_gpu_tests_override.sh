#!/bin/bash
set -euo pipefail

EXP_ID="${1:?Usage: $0 <experiment_id>}"

echo "Verifying experiment: $EXP_ID"

exp_json=$(beaker experiment get "$EXP_ID" --format json)
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to get experiment $EXP_ID"
    exit 1
fi

description=$(echo "$exp_json" | jq -r '.[0].description // ""')
if ! echo "$description" | grep -qi "GPU tests"; then
    echo "ERROR: Experiment description must contain 'GPU tests'"
    echo "Found description: $description"
    exit 1
fi

exit_code=$(echo "$exp_json" | jq -r '.[0].jobs[0].status.exitCode // "null"')
if [ "$exit_code" != "0" ]; then
    echo "ERROR: Experiment exit code must be 0, got: $exit_code"
    exit 1
fi

echo ""
echo "=========================================="
echo "GPU_TESTS override verified successfully!"
echo "Experiment: $EXP_ID"
echo "Description: $description"
echo "Exit code: $exit_code"
echo "=========================================="
echo ""
