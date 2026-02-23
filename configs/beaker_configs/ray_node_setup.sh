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

# Ray can leave stale metadata in /tmp/ray briefly after stop; clear it before start.
rm -rf /tmp/ray/session_* /tmp/ray/ray_current_cluster 2>/dev/null || true

# Give process teardown a short window to finish before starting a new head.
for _ in 1 2 3 4 5; do
    if ! pgrep -fa "raylet|gcs_server|dashboard|monitor.py" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if [ "$BEAKER_REPLICA_RANK" == "0" ]; then
    echo "Starting Ray head node"
    for attempt in 1 2 3; do
        if ray start --head --port=$RAY_NODE_PORT --dashboard-host=0.0.0.0; then
            export RAY_ADDRESS="127.0.0.1:${RAY_NODE_PORT}"
            break
        fi
        echo "[ray_node_setup] Ray head start failed on attempt ${attempt}; retrying after cleanup"
        ray stop --force >/dev/null 2>&1 || true
        rm -rf /tmp/ray/session_* /tmp/ray/ray_current_cluster 2>/dev/null || true
        sleep 2
        if [ "$attempt" -eq 3 ]; then
            echo "[ray_node_setup] Failed to start Ray head after ${attempt} attempts"
            exit 1
        fi
    done
else
    echo "Starting Ray worker node $BEAKER_REPLICA_RANK"
    export RAY_ADDRESS="${BEAKER_LEADER_REPLICA_IP}:${RAY_NODE_PORT}"
    # Start worker without --block so we can control lifecycle and exit code.
    ray start --address="${RAY_ADDRESS}" --dashboard-host=0.0.0.0

    cleanup() {
        echo "[ray_node_setup] Cleanup: stopping Ray worker and exiting 0"
        ray stop --force >/dev/null 2>&1 || true
        trap - TERM INT HUP EXIT
        exit 0
    }

    trap cleanup TERM INT HUP EXIT

    echo "[ray_node_setup] Monitoring Ray head at ${RAY_ADDRESS}"
    # Poll head availability. Exit 0 when head is gone.
    while true; do
        if ! ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
            echo "[ray_node_setup] Head is unreachable. Stopping worker and exiting 0."
            cleanup
        fi
        sleep 5
    done
fi
