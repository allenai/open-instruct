#!/bin/bash
# Wait for a Beaker experiment to finalize and report its result.
#
# Usage: wait_for_beaker_experiment.sh <experiment_id> [timeout]
#
# Exit codes:
#   0  every task and replica finished with exit code 0
#   1  a task or replica finished with a numeric non-zero exit code (a real failure)
#   2  the experiment did not finalize within the timeout, or the wait itself failed
#      (for example a Beaker API error); it is stopped so an abandoned CI job does
#      not keep a place in the Beaker queue
#   3  the experiment finalized but a task or replica has no exit code (preempted or
#      cancelled), or its result could not be read
set -euo pipefail

EXP_ID="${1:?Usage: $0 <experiment_id> [timeout]}"
TIMEOUT="${2:-18m}"

echo "Waiting up to ${TIMEOUT} for https://beaker.org/ex/${EXP_ID}"
if ! beaker experiment await "$EXP_ID" 0 finalized --index --timeout "$TIMEOUT" --interval 30s; then
    echo "Experiment ${EXP_ID} did not finalize within ${TIMEOUT}; stopping it."
    beaker experiment stop "$EXP_ID" || true
    exit 2
fi

# An experiment holds one job per attempt of each task and replica; a preempted job may
# have been retried. Judge the most recent attempt of every task and replica, so a
# successful retry is not read as a failure and a failed replica is not overlooked.
if ! exit_codes=$(beaker experiment get "$EXP_ID" --format json | jq -r '
    .[0].jobs
    | group_by([.execution.task, (.execution.replicaRank // 0)])
    | map(max_by(.status.created) | .status.exitCode // "null")
    | .[]'); then
    echo "Could not read the result of experiment ${EXP_ID}"
    exit 3
fi
if [ -z "$exit_codes" ]; then
    echo "Experiment ${EXP_ID} has no jobs"
    exit 3
fi

failed_codes=$(grep -E '^[0-9]+$' <<< "$exit_codes" | grep -vx 0 | sort -u | paste -sd, - || true)
if [ -n "$failed_codes" ]; then
    echo "Experiment ${EXP_ID} failed with exit code ${failed_codes}"
    exit 1
fi
if grep -qvx 0 <<< "$exit_codes"; then
    echo "Experiment ${EXP_ID} finalized without an exit code (preempted or cancelled)"
    exit 3
fi
echo "Experiment ${EXP_ID} succeeded"
exit 0
