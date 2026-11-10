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

    # A task whose spec has no `replicas` gets no BEAKER_REPLICA_RANK, so it reaches this
    # branch with no leader to join. Fail loudly rather than reporting success.
    if [ -z "${BEAKER_LEADER_REPLICA_HOSTNAME}" ]; then
        echo "[ray_node_setup] BEAKER_LEADER_REPLICA_HOSTNAME is empty, so there is no Ray head to join." >&2
        echo "[ray_node_setup] The task spec is probably missing 'replicas'. Exiting 1." >&2
        exit 1
    fi

    export RAY_ADDRESS="${BEAKER_LEADER_REPLICA_IP}:${RAY_NODE_PORT}"

    cleanup() {
        local code="${1:-0}"
        echo "[ray_node_setup] Cleanup: stopping Ray worker and exiting ${code}"
        ray stop --force >/dev/null 2>&1 || true
        trap - TERM INT HUP EXIT
        exit "${code}"
    }

    trap 'cleanup 0' TERM INT HUP EXIT

    # Start worker without --block so we can control lifecycle and exit code.
    if ! ray start --address="${RAY_ADDRESS}" --dashboard-host=0.0.0.0; then
        echo "[ray_node_setup] Failed to join the Ray head at ${RAY_ADDRESS}. Exiting 1." >&2
        cleanup 1
    fi

    # Confirm the join took hold. `ray start` can return 0 without this node being
    # registered with the head, and a worker that never joined must not report success.
    if ! ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
        echo "[ray_node_setup] Started, but the head at ${RAY_ADDRESS} is not reachable. Exiting 1." >&2
        cleanup 1
    fi

    echo "[ray_node_setup] Monitoring Ray head at ${RAY_ADDRESS}"
    # Past this point a head that disappears is the normal end of a run, so exit 0.
    while true; do
        if ! ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
            echo "[ray_node_setup] Head is unreachable. Stopping worker and exiting 0."
            cleanup 0
        fi
        sleep 5
    done
fi
