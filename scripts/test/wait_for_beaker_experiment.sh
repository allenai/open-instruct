#!/bin/bash
# Wait for a Beaker experiment to finalize and report its result.
#
# Usage: wait_for_beaker_experiment.sh <experiment_id> [timeout]
#
# Exit codes:
#   0  the experiment finalized with exit code 0
#   1  the experiment finalized with a numeric non-zero exit code (a real failure)
#   2  the experiment did not finalize within the timeout; it is stopped so an
#      abandoned CI job does not keep a place in the Beaker queue
#   3  the experiment finalized without an exit code (preempted or cancelled)
set -euo pipefail

EXP_ID="${1:?Usage: $0 <experiment_id> [timeout]}"
TIMEOUT="${2:-18m}"

echo "Waiting up to ${TIMEOUT} for https://beaker.org/ex/${EXP_ID}"
if ! beaker experiment await "$EXP_ID" 0 finalized --index --timeout "$TIMEOUT" --interval 30s; then
    echo "Experiment ${EXP_ID} did not finalize within ${TIMEOUT}; stopping it."
    beaker experiment stop "$EXP_ID" || true
    exit 2
fi

# Read the most recent attempt: a preempted job may have been retried.
exit_code=$(beaker experiment get "$EXP_ID" --format json | jq -r '.[0].jobs[-1].status.exitCode // "null"')
if [ "$exit_code" = "0" ]; then
    echo "Experiment ${EXP_ID} succeeded"
    exit 0
fi
if [[ "$exit_code" =~ ^[0-9]+$ ]]; then
    echo "Experiment ${EXP_ID} failed with exit code ${exit_code}"
    exit 1
fi
echo "Experiment ${EXP_ID} finalized without an exit code (preempted or cancelled)"
exit 3
