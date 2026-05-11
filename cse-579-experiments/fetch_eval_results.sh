#!/bin/bash
# Fetch eval results from a list of Beaker lmeval experiments into
# cse-579-experiments/results/<run_dir>/<task>/.
#
# Usage:
#   bash cse-579-experiments/fetch_eval_results.sh <run_dir> <exp_id> [<exp_id> ...]
#
# Example:
#   bash cse-579-experiments/fetch_eval_results.sh \
#     lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant_step_1000 \
#     01KRCA5XQS5JV19DV7GHM0T7PE 01KRCA5YJA5W0VSNZ90BQKGF75 ...
#
# For each experiment we save:
#   metrics.json            (headline scores)
#   task-*-metrics.json     (per-task detail)
#   length_stats.json       (computed: continuation char-length distribution)
# We deliberately skip predictions.jsonl (multi-MB) and requests.jsonl
# (multi-MB); fetch those manually from Beaker if needed for deeper analysis.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <run_dir> <exp_id> [<exp_id> ...]"
    exit 1
fi

RUN_DIR="$1"
shift

REPO_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
DEST_ROOT="$REPO_ROOT/cse-579-experiments/results/$RUN_DIR"
mkdir -p "$DEST_ROOT"

for EXP_ID in "$@"; do
    echo "==> $EXP_ID"
    EXP_JSON=$(beaker experiment get "$EXP_ID" --format json 2>/dev/null)
    if [ -z "$EXP_JSON" ]; then
        echo "    (could not fetch experiment metadata, skipping)"
        continue
    fi
    EXIT_CODE=$(echo "$EXP_JSON" | jq -r '.[0].jobs[0].status.exitCode // "none"')
    RESULT_ID=$(echo "$EXP_JSON" | jq -r '.[0].jobs[0].result.beaker // "none"')
    NAME=$(echo "$EXP_JSON" | jq -r '.[0].name')

    if [ "$EXIT_CODE" != "0" ]; then
        echo "    skipping (exitCode=$EXIT_CODE, name=$NAME)"
        continue
    fi
    if [ "$RESULT_ID" = "none" ]; then
        echo "    skipping (no result dataset)"
        continue
    fi

    # Task short name derived from the Beaker experiment name:
    #   ianm/lmeval-<model>-on-<task>-<hash>  =>  <task>
    TASK=$(echo "$NAME" | sed -E 's/.*-on-//; s/-[0-9a-f]{10}$//')
    TASK_DIR="$DEST_ROOT/$TASK"
    mkdir -p "$TASK_DIR"

    # Always pull the small JSON; pull predictions only into a tmp scratch so we
    # can derive length stats without committing the multi-MB file.
    beaker dataset fetch "$RESULT_ID" --prefix metrics.json -o "$TASK_DIR" --quiet >/dev/null
    beaker dataset fetch "$RESULT_ID" --prefix metrics-all.jsonl -o "$TASK_DIR" --quiet >/dev/null
    # Per-task metrics (filename includes the task name with a 'task-NNN-' prefix)
    beaker dataset fetch "$RESULT_ID" --prefix "task-" -o "$TASK_DIR" --quiet >/dev/null 2>&1 || true
    # Remove the predictions/requests/inputs jsonl (large, not committable)
    find "$TASK_DIR" -maxdepth 1 -type f \
         \( -name '*predictions.jsonl' -o -name '*requests.jsonl' -o -name '*recorded-inputs.jsonl' \) \
         -print -delete >/dev/null 2>&1 || true

    # Derive length stats from predictions before we delete them.
    SCRATCH=$(mktemp -d)
    beaker dataset fetch "$RESULT_ID" --prefix "task-" -o "$SCRATCH" --quiet >/dev/null
    PRED=$(find "$SCRATCH" -name '*predictions.jsonl' | head -1)
    if [ -n "$PRED" ] && [ -s "$PRED" ]; then
        jq -s '
            map(.model_output[0].continuation | length) as $L
            | {
                n: ($L | length),
                min: ($L | min),
                max: ($L | max),
                mean: (($L | add) / ($L | length)),
                median: ($L | sort | .[(length/2|floor)]),
                p10: ($L | sort | .[(length*0.1|floor)]),
                p90: ($L | sort | .[(length*0.9|floor)])
              }' "$PRED" > "$TASK_DIR/length_stats.json"
        echo "    saved metrics + length stats -> $TASK_DIR"
    else
        echo "    saved metrics (no predictions found) -> $TASK_DIR"
    fi
    rm -rf "$SCRATCH"
done

echo
echo "Done. Saved under: $DEST_ROOT"
