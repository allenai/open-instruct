#!/bin/bash
set -euo pipefail

EVENT_NAME="${1:-}"
HEAD_REPO="${2:-}"
BASE_REPO="${3:-}"
PR_BODY="${4:-}"

# Results go to GITHUB_ENV for later steps in this job and to GITHUB_OUTPUT so
# the separate wait job can read them. The last value written wins.
set_result() {
    echo "$1=$2" >> "$GITHUB_ENV"
    if [ -n "${GITHUB_OUTPUT:-}" ]; then
        echo "$(echo "$1" | tr '[:upper:]' '[:lower:]')=$2" >> "$GITHUB_OUTPUT"
    fi
}

set_result RUN_GPU_TESTS true
set_result SKIP_REASON ""
set_result GPU_TESTS_EXP_ID ""

if [ "$EVENT_NAME" = "merge_group" ]; then
    echo "Merge queue: running GPU tests"
    exit 0
fi

if [ "$HEAD_REPO" != "$BASE_REPO" ]; then
    set_result RUN_GPU_TESTS "false"
    set_result SKIP_REASON "fork"
    echo ""
    echo "=========================================="
    echo "Skipping GPU tests for fork PR"
    echo "=========================================="
    echo ""
    echo "This PR is from a fork ($HEAD_REPO), and secrets are not available."
    echo "GPU tests will run automatically when this PR enters the merge queue."
    echo ""
    exit 0
fi

if echo "$PR_BODY" | grep -qE 'GPU_TESTS=bypass'; then
    set_result RUN_GPU_TESTS "false"
    set_result SKIP_REASON "bypass"
    echo ""
    echo "=========================================="
    echo "GPU_TESTS=bypass found, skipping GPU tests"
    echo "=========================================="
    echo ""
    exit 0
fi

if echo "$PR_BODY" | grep -qE 'GPU_TESTS=\[[^]]+\]\([^)]+\)'; then
    EXP_ID=$(echo "$PR_BODY" | grep -oP 'GPU_TESTS=\[\K[^]]+(?=\]\([^)]+\))')
    set_result GPU_TESTS_EXP_ID "$EXP_ID"
    set_result RUN_GPU_TESTS "false"
    set_result SKIP_REASON "override"
    echo ""
    echo "=========================================="
    echo "Found GPU_TESTS override: $EXP_ID"
    echo "=========================================="
    echo ""
fi
