#!/bin/bash
set -euo pipefail

EVENT_NAME="${1:-}"
HEAD_REPO="${2:-}"
BASE_REPO="${3:-}"
PR_BODY="${4:-}"

echo "RUN_GPU_TESTS=true" >> "$GITHUB_ENV"
echo "SKIP_REASON=" >> "$GITHUB_ENV"
echo "GPU_TESTS_EXP_ID=" >> "$GITHUB_ENV"

if [ "$EVENT_NAME" = "merge_group" ]; then
    echo "Merge queue: running GPU tests"
    exit 0
fi

if [ "$HEAD_REPO" != "$BASE_REPO" ]; then
    echo "RUN_GPU_TESTS=false" >> "$GITHUB_ENV"
    echo "SKIP_REASON=fork" >> "$GITHUB_ENV"
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
    echo "RUN_GPU_TESTS=false" >> "$GITHUB_ENV"
    echo "SKIP_REASON=bypass" >> "$GITHUB_ENV"
    echo ""
    echo "=========================================="
    echo "GPU_TESTS=bypass found, skipping GPU tests"
    echo "=========================================="
    echo ""
    exit 0
fi

if echo "$PR_BODY" | grep -qE 'GPU_TESTS=\[[^\]]+\]\([^)]+\)'; then
    EXP_ID=$(echo "$PR_BODY" | grep -oP 'GPU_TESTS=\[\K[^\]]+')
    echo "GPU_TESTS_EXP_ID=$EXP_ID" >> "$GITHUB_ENV"
    echo "RUN_GPU_TESTS=false" >> "$GITHUB_ENV"
    echo "SKIP_REASON=override" >> "$GITHUB_ENV"
    echo ""
    echo "=========================================="
    echo "Found GPU_TESTS override: $EXP_ID"
    echo "=========================================="
    echo ""
fi
