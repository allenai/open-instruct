#!/bin/bash
# Launch the GPU tests on Beaker and wait for the result in one step.
# CI runs the two halves as separate jobs (see .github/workflows/tests.yml);
# this wrapper is for running them by hand.
set -euo pipefail

GIT_SHA="${1:?Usage: $0 <git_sha>}"

output=$(bash scripts/test/launch_gpu_tests_on_beaker.sh "$GIT_SHA" 2>&1)
echo "$output"
exp_id=$(echo "$output" | grep -oP 'Launched GPU tests: https://beaker.org/ex/\K[^\s]+' | head -1)

# Stop the experiment if we exit before it finishes (for example Ctrl-C), so it does
# not take a GPU later for a result nobody reads. A timeout is handled by the wait script.
finished=false
stop_unfinished() {
    if [ "$finished" != "true" ]; then
        echo "Stopping unfinished experiment $exp_id"
        beaker experiment stop "$exp_id" || true
    fi
}
trap stop_unfinished EXIT
trap 'exit 130' INT TERM

result=0
bash scripts/test/wait_for_beaker_experiment.sh "$exp_id" 40m || result=$?
finished=true
exit "$result"
