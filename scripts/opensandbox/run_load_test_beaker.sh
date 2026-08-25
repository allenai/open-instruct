#!/bin/bash
# In-job runner for the OpenSandbox scale test (see launch_load_test_beaker.sh).
#
# Runs the load test as TWO side-by-side 1024-worker processes with distinct
# app names — simulating two independent training jobs hitting the service —
# and exits nonzero if either process fails its success criteria. Sweeps both
# app tags with the janitor afterwards regardless of outcome.
#
# Tunables (env, all optional):
#   LOAD_TEST_WORKERS_PER_PROC (default 1024)
#   LOAD_TEST_DURATION_S       (default 3600)
#   LOAD_TEST_CPU              (default 1.0)  — sandbox CPU reservation

set -uo pipefail

: "${OPEN_SANDBOX_API_KEY:?OPEN_SANDBOX_API_KEY must be set (Beaker secret)}"
: "${SWERL_OPENSANDBOX_DOMAIN:?SWERL_OPENSANDBOX_DOMAIN must be set}"

WORKERS="${LOAD_TEST_WORKERS_PER_PROC:-1024}"
DURATION="${LOAD_TEST_DURATION_S:-3600}"
CPU="${LOAD_TEST_CPU:-1.0}"

ulimit -n 65536 2>/dev/null || echo "warning: could not raise fd limit ($(ulimit -n) available)"

run_one() {
    local tag="$1"
    python scripts/opensandbox/load_test.py \
        --workers "$WORKERS" \
        --duration-s "$DURATION" \
        --episode-s 240 \
        --execs-per-episode 8 \
        --ramp-s 420 \
        --cpu "$CPU" \
        --mem-limit 4g \
        --sandbox-lifetime-s 1800 \
        --app-name "loadtest-$tag" 2>&1 | grep -v "HTTP Request" > "/tmp/loadtest-$tag.log"
    return "${PIPESTATUS[0]}"
}

echo "=== OpenSandbox scale test: 2 x $WORKERS workers, ${DURATION}s, cpu=$CPU ==="
run_one a & PID_A=$!
run_one b & PID_B=$!
wait "$PID_A"; STATUS_A=$?
wait "$PID_B"; STATUS_B=$?

for tag in a b; do
    echo; echo "########## process $tag: final report ##########"
    # The summary block plus the last few reporter lines.
    grep "active=" "/tmp/loadtest-$tag.log" | tail -3
    sed -n '/LOAD TEST SUMMARY/,$p' "/tmp/loadtest-$tag.log"
done

echo; echo "=== janitor sweep ==="
for tag in a b; do
    bash scripts/opensandbox/cleanup_opensandbox_sandboxes.sh "loadtest-$tag" || true
done

echo; echo "process a exit=$STATUS_A, process b exit=$STATUS_B"
[ "$STATUS_A" -eq 0 ] && [ "$STATUS_B" -eq 0 ]
