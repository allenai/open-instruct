#!/bin/bash
# Fetch eval results from a list of Beaker lmeval experiments into
# cse-579-experiments/results/<run>/step_<step>/<task>/.
#
# Usage:
#   bash cse-579-experiments/fetch_eval_results.sh <run> <step> <exp_id> [<exp_id> ...]
#
# Example:
#   bash cse-579-experiments/fetch_eval_results.sh \
#     lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant 1000 \
#     01KRCA5XQS5JV19DV7GHM0T7PE 01KRCA5YJA5W0VSNZ90BQKGF75 ...
#
# For each experiment we save:
#   metrics.json            (headline scores)
#   task-*-metrics.json     (per-task detail)
#   length_stats.json       (computed: continuation char-length distribution)
# We deliberately skip predictions.jsonl (multi-MB) and requests.jsonl
# (multi-MB); fetch those manually from Beaker if needed for deeper analysis.

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <run> <step> <exp_id> [<exp_id> ...]"
    exit 1
fi

RUN="$1"
STEP="$2"
shift 2

REPO_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
DEST_ROOT="$REPO_ROOT/cse-579-experiments/results/$RUN/step_$STEP"
mkdir -p "$DEST_ROOT"

for EXP_ID in "$@"; do
    echo "==> $EXP_ID"
    EXP_JSON=$(beaker experiment get "$EXP_ID" --format json 2>/dev/null)
    if [ -z "$EXP_JSON" ]; then
        echo "    (could not fetch experiment metadata, skipping)"
        continue
    fi
    NAME=$(echo "$EXP_JSON" | jq -r '.[0].name')

    # Pick the latest job that succeeded (preemption + retry is common at normal
    # priority — looking only at jobs[0] would consume the dead first attempt).
    SUCCESSFUL_JOB=$(echo "$EXP_JSON" | jq '.[0].jobs | map(select(.status.exitCode == 0)) | sort_by(.status.exited) | last // null')
    if [ "$SUCCESSFUL_JOB" = "null" ] || [ -z "$SUCCESSFUL_JOB" ]; then
        TRIES=$(echo "$EXP_JSON" | jq -r '.[0].jobs | map("[exit=\(.status.exitCode // "-")@\(.status.exited // "running")]") | join(", ")')
        echo "    skipping (no successful job among: $TRIES, name=$NAME)"
        continue
    fi
    RESULT_ID=$(echo "$SUCCESSFUL_JOB" | jq -r '.result.beaker // "none"')
    if [ "$RESULT_ID" = "none" ]; then
        echo "    skipping (successful job has no result dataset, name=$NAME)"
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

    # Derive length stats from each subtask's predictions. ifbench is the case
    # where one Beaker experiment expands into multiple subtask predictions
    # files; we want one length_stats.json per subtask (paired with its
    # task-NNN-{taskname}-metrics.json).
    SCRATCH=$(mktemp -d)
    beaker dataset fetch "$RESULT_ID" --prefix "task-" -o "$SCRATCH" --quiet >/dev/null
    found_any=0
    for PRED in "$SCRATCH"/task-*-predictions.jsonl; do
        [ -e "$PRED" ] || continue
        [ -s "$PRED" ] || continue
        STEM=$(basename "$PRED" -predictions.jsonl)        # e.g. task-000-aime
        TASK_METRICS="$TASK_DIR/${STEM}-metrics.json"
        if [ ! -f "$TASK_METRICS" ]; then
            echo "    WARNING: no metrics file for $STEM; skipping length stats"
            continue
        fi
        uv run python "$REPO_ROOT/cse-579-experiments/compute_length_stats.py" \
            "$PRED" "$TASK_METRICS" "$TASK_DIR/${STEM}-length_stats.json"
        found_any=1
    done
    if [ "$found_any" -eq 1 ]; then
        echo "    saved metrics + per-subtask length stats -> $TASK_DIR"
    else
        echo "    saved metrics (no predictions found) -> $TASK_DIR"
    fi
    rm -rf "$SCRATCH"
done

echo
echo "Done. Saved under: $DEST_ROOT"
