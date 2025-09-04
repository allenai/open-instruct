#!/bin/bash
CURRENT_DATETIME=$(python -c "import datetime; import pytz; print(datetime.datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%m%d%y_%H%M%S'))")
export CURRENT_DATETIME
export PYTHONPATH="$REPO_PATH"
export PATH="/root/.local/bin:$PATH"
# We need to set NCCL_CUMEM_ENABLE=0 for performance reasons; see: 
# https://github.com/vllm-project/vllm/issues/5723#issuecomment-2554389656
export NCCL_CUMEM_ENABLE=0

# Function to gracefully shutdown Ray
graceful_ray_shutdown() {
    echo "Received signal, shutting down Ray gracefully..."
    
    # Try graceful shutdown first
    timeout 30 ray stop
    exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "Graceful shutdown failed, forcing Ray shutdown..."
        timeout 15 ray stop --force
        force_exit_code=$?
        
        if [ $force_exit_code -ne 0 ]; then
            echo "Force shutdown also failed, killing Ray processes..."
            pkill -f "ray::" || true
            pkill -f "raylet" || true
            pkill -f "gcs_server" || true
        fi
    fi
    
    echo "Ray shutdown complete"
    exit 0
}

# Set up signal handlers for graceful shutdown
trap graceful_ray_shutdown SIGTERM SIGINT SIGQUIT

echo "CURRENT_DATETIME=$CURRENT_DATETIME"
echo "PYTHONPATH=$PYTHONPATH"
echo "PATH=$PATH"

# python3 -c "import os, ray; print(os.path.dirname(ray.__file__))"

BEAKER_LEADER_REPLICA_IP=$(getent hosts "${BEAKER_LEADER_REPLICA_HOSTNAME}" | awk '{print $1}')

RAY_NODE_PORT=8888
ray stop --force

if [ "$BEAKER_REPLICA_RANK" == "0" ]; then
    echo "Starting Ray head node"
    ray start --head --port=$RAY_NODE_PORT --dashboard-host=0.0.0.0
else
    echo "Starting Ray worker node $BEAKER_REPLICA_RANK"
    # For worker nodes, we need to handle the case where the head node dies
    ray start --address="${BEAKER_LEADER_REPLICA_IP}:${RAY_NODE_PORT}" --block --dashboard-host=0.0.0.0 &
    RAY_PID=$!
    
    # Wait for Ray process and handle graceful exit if it fails due to head node death
    wait $RAY_PID
    ray_exit_code=$?
    
    # If Ray worker exits due to head node failure, clean up and exit gracefully
    if [ $ray_exit_code -ne 0 ]; then
        echo "Ray worker exited with code $ray_exit_code (likely due to head node failure)"
        graceful_ray_shutdown
    fi
fi