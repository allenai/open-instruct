#!/bin/bash
# Find all lmeval experiments matching a model prefix, group by step, and
# fetch results into per-step result directories.
#
# Usage:
#   bash cse-579-experiments/fetch_all_steps.sh <model_prefix> <results_run_prefix>
#
# Example:
#   bash cse-579-experiments/fetch_all_steps.sh \
#     baseline_think_run_4b_base_mixed_32k__1__1776217615 \
#     baseline_think_run_4b_base_mixed_32k
#
# This will look in ai2/olmo-instruct for experiments named like
# 'lmeval-<model_prefix>_step_NNN-on-...' and fetch each step into
# cse-579-experiments/results/<results_run_prefix>/step_NNN/.

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_prefix> <results_run_prefix>" >&2
    exit 1
fi

MODEL_PREFIX="$1"
RESULTS_RUN_PREFIX="$2"
REPO_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
FETCHER="$REPO_ROOT/cse-579-experiments/fetch_eval_results.sh"

echo "Finding lmeval experiments matching '$MODEL_PREFIX' in ai2/olmo-instruct..."
LIST=$(beaker workspace experiments ai2/olmo-instruct --text "$MODEL_PREFIX" --format json 2>/dev/null)
TOTAL=$(echo "$LIST" | jq 'length')
echo "  found $TOTAL experiments"

# Group exp ids by step using a temp file (works on macOS bash 3.2).
SCRATCH=$(mktemp -d)
echo "$LIST" | jq -r --arg prefix "$MODEL_PREFIX" '
  .[] | select(.name | test("lmeval-" + $prefix + "_step_[0-9]+-on-"))
      | (.name | capture("_step_(?<s>[0-9]+)-on-").s) + "\t" + .id' \
  > "$SCRATCH/pairs.tsv"

if [ ! -s "$SCRATCH/pairs.tsv" ]; then
    echo "  no matching experiments found"
    rm -rf "$SCRATCH"
    exit 0
fi

STEPS=$(awk -F'\t' '{print $1}' "$SCRATCH/pairs.tsv" | sort -un)
for step in $STEPS; do
    IDS=$(awk -F'\t' -v s="$step" '$1 == s { print $2 }' "$SCRATCH/pairs.tsv" | tr '\n' ' ')
    N=$(echo "$IDS" | wc -w | tr -d ' ')
    echo
    echo "===== step $step ($N experiments) -> $RESULTS_RUN_PREFIX/step_$step ====="
    # shellcheck disable=SC2086  # intentional word-splitting on $IDS
    bash "$FETCHER" "$RESULTS_RUN_PREFIX" "$step" $IDS 2>&1 | grep -E "^==|^    saved|skipping" || true
done
rm -rf "$SCRATCH"
