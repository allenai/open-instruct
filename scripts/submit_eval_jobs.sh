#!/usr/bin/env bash
# Submit an olmo-eval Beaker experiment.
#
# Usage:
#   scripts/submit_eval_jobs.sh -m /weka/.../checkpoint -t aime_2025:pass_at_32 \
#       -c h100 -p urgent --preemptible \
#       -w ai2/open-instruct-dev -B ai2/oe-adapt --gpus 1
#
# All flags pass through to `olmo-eval beaker launch`. Run
# `scripts/submit_eval_jobs.sh --help` to see them.
#
# Override the olmo-eval-internal git ref via OLMO_EVAL_REF=<branch|sha|tag>
# (default: main).
set -euo pipefail

REPO_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
CLONE="$REPO_ROOT/olmo-eval-internal"
REF="${OLMO_EVAL_REF:-main}"

if [ ! -d "$CLONE/.git" ]; then
    git clone --depth=1 https://github.com/allenai/olmo-eval-internal.git "$CLONE"
fi
git -C "$CLONE" fetch --depth=1 origin "$REF"
git -C "$CLONE" checkout FETCH_HEAD

cd "$CLONE"
exec uv run olmo-eval beaker launch -y \
    --harness default \
    -o provider.kind=vllm_server \
    -o provider.trust_remote_code=true \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    "$@"
