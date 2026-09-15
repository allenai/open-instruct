#!/bin/bash

# Build the code-patch Beaker dataset that the Qwen3.5 math launch scripts
# overlay onto the pinned training image (mounted at /patch and copied into
# /stage/open_instruct/ before training). Run from a clean, committed checkout
# so the dataset name records the exact source revision.
#
# Usage: scripts/general_agent/terminal/rl/make_math_patch_dataset.sh [workspace]
set -euo pipefail

WORKSPACE="${1:-${WORKSPACE:-ai2/olmo-instruct}}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
FILES=(data_loader.py ground_truth_utils.py grpo_fast.py grpo_utils.py actor_manager.py)

if [[ -n "$(git -C "$REPO" status --porcelain -- open_instruct)" ]]; then
    echo "open_instruct/ has uncommitted changes; commit them so the patch is reproducible" >&2
    exit 1
fi
SHA="$(git -C "$REPO" rev-parse --short HEAD)"
STAGE="$(mktemp -d)/open_instruct"
mkdir -p "$STAGE"
for f in "${FILES[@]}"; do
    cp "$REPO/open_instruct/$f" "$STAGE/"
done
NAME="qwen35-math-code-patch-$SHA"
echo "Creating Beaker dataset $NAME in $WORKSPACE from $(dirname "$STAGE")"
beaker dataset create --workspace "$WORKSPACE" --name "$NAME" "$(dirname "$STAGE")"
