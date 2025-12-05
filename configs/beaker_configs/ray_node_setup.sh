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
mkdir -p "$HOME/.triton/autotune"

echo "Cleaning up any existing Ray processes..."
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "gcs_server" 2>/dev/null || true
pkill -9 -f "raylet" 2>/dev/null || true
pkill -9 -f "redis-server" 2>/dev/null || true
pkill -9 -f "plasma_store" 2>/dev/null || true
pkill -9 -f "log_monitor" 2>/dev/null || true
pkill -9 -f "monitor.py" 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 3

rm -rf /tmp/ray* 2>/dev/null || true
rm -rf /dev/shm/* 2>/dev/null || true
rm -rf ~/.ray 2>/dev/null || true
rm -rf /run/user/*/ray* 2>/dev/null || true
sleep 2

RAY_TEMP_DIR="/tmp/r_$(date +%s)"
mkdir -p "$RAY_TEMP_DIR"
export RAY_TMPDIR="$RAY_TEMP_DIR"

if [ "$BEAKER_REPLICA_RANK" == "0" ]; then
    echo "Starting Ray head node with temp dir: $RAY_TEMP_DIR"
    ray start --head --port=$RAY_NODE_PORT --dashboard-host=0.0.0.0 --temp-dir="$RAY_TEMP_DIR" --disable-usage-stats
else
    echo "Starting Ray worker node $BEAKER_REPLICA_RANK with temp dir: $RAY_TEMP_DIR"
    export RAY_ADDRESS="${BEAKER_LEADER_REPLICA_IP}:${RAY_NODE_PORT}"
    ray start --address="${RAY_ADDRESS}" --dashboard-host=0.0.0.0 --temp-dir="$RAY_TEMP_DIR"

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
