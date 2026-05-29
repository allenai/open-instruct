#!/bin/bash
export CURRENT_DATETIME=$(python -c "import datetime; import pytz; print(datetime.datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%m%d%y_%H%M%S'))")
export PYTHONPATH=$REPO_PATH
export PATH="/root/.local/bin:$PATH"
# We need to set NCCL_CUMEM_ENABLE=0 for performance reasons; see:
# https://github.com/vllm-project/vllm/issues/5723#issuecomment-2554389656
export NCCL_CUMEM_ENABLE=0


echo CURRENT_DATETIME=$CURRENT_DATETIME
echo PYTHONPATH=$PYTHONPATH
echo PATH=$PATH

# python3 -c "import os, ray; print(os.path.dirname(ray.__file__))"

BEAKER_LEADER_REPLICA_IP=$(getent hosts ${BEAKER_LEADER_REPLICA_HOSTNAME} | awk '{print $1}')

RAY_NODE_PORT=8888
mkdir -p "$HOME/.triton/autotune"  # Create Triton autotune cache directory to silence warnings
ray stop --force

if [ "$BEAKER_REPLICA_RANK" == "0" ]; then
    echo "Starting Ray head node"
    ray start --head --port=$RAY_NODE_PORT --dashboard-host=0.0.0.0
else
    echo "Starting Ray worker node $BEAKER_REPLICA_RANK"
    export RAY_ADDRESS="${BEAKER_LEADER_REPLICA_IP}:${RAY_NODE_PORT}"
    # Start worker without --block so we can control lifecycle and exit code.
    ray start --address="${RAY_ADDRESS}" --dashboard-host=0.0.0.0

    cleanup() {
        exit_code="${1:-1}"
        if [ "$exit_code" -eq 0 ]; then
            echo "[ray_node_setup] Worker cleanup requested exit 0; forcing exit 1."
            exit_code=1
        fi
        echo "[ray_node_setup] Cleanup: stopping Ray worker and exiting ${exit_code}"
        ray stop --force >/dev/null 2>&1 || true
        trap - TERM INT HUP EXIT
        exit "$exit_code"
    }

    trap 'cleanup 143' TERM
    trap 'cleanup 130' INT
    trap 'cleanup 129' HUP
    trap 'cleanup 1' EXIT

    RAY_HEAD_MONITOR_INTERVAL_S="${RAY_HEAD_MONITOR_INTERVAL_S:-5}"
    RAY_HEAD_MONITOR_MAX_MISSES="${RAY_HEAD_MONITOR_MAX_MISSES:-6}"
    ray_head_monitor_misses=0

    echo "[ray_node_setup] Monitoring Ray head at ${RAY_ADDRESS}"
    # Poll head availability. Workers should never report success on shutdown.
    # Require consecutive misses so transient Ray status hiccups don't kill the worker.
    while true; do
        if ! ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
            ray_head_monitor_misses=$((ray_head_monitor_misses + 1))
            echo "[ray_node_setup] Head status check failed (${ray_head_monitor_misses}/${RAY_HEAD_MONITOR_MAX_MISSES})."
            if [ "$ray_head_monitor_misses" -ge "$RAY_HEAD_MONITOR_MAX_MISSES" ]; then
                echo "[ray_node_setup] Head is unreachable. Stopping worker and exiting 1."
                cleanup 1
            fi
        else
            if [ "$ray_head_monitor_misses" -gt 0 ]; then
                echo "[ray_node_setup] Head status recovered after ${ray_head_monitor_misses} failed checks."
            fi
            ray_head_monitor_misses=0
        fi
        sleep "$RAY_HEAD_MONITOR_INTERVAL_S"
    done
fi
