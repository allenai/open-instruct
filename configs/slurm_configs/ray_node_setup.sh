#!/bin/bash
# Ray node setup script for Slurm multi-node jobs
# Adapted from configs/beaker_configs/ray_node_setup.sh
#
# Expected environment variables (set by Slurm or parent script):
#   MASTER_ADDR - IP/hostname of the head node
#   SLURM_NODEID - Node rank (0 for head, 1+ for workers)
#   GPUS_PER_NODE - Number of GPUs per node (optional, defaults to auto-detect)
#
# Usage: source this script or run it on each node via srun

set -e

export CURRENT_DATETIME=$(python -c "import datetime; print(datetime.datetime.now().strftime('%m%d%y_%H%M%S'))")
export PYTHONPATH=${PYTHONPATH:-$(pwd)}

# Performance settings
# We need to set NCCL_CUMEM_ENABLE=0 for performance reasons; see:
# https://github.com/vllm-project/vllm/issues/5723#issuecomment-2554389656
export NCCL_CUMEM_ENABLE=0

echo "=== Ray Node Setup ==="
echo "CURRENT_DATETIME=$CURRENT_DATETIME"
echo "HOSTNAME=$(hostname)"
echo "SLURM_NODEID=${SLURM_NODEID:-not set}"
echo "MASTER_ADDR=${MASTER_ADDR:-not set}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE:-auto}"

RAY_NODE_PORT=${RAY_PORT:-6379}

# Create Triton autotune cache directory to silence warnings
mkdir -p "$HOME/.triton/autotune" 2>/dev/null || true

# Stop any existing Ray processes
ray stop --force 2>/dev/null || true

if [ "${SLURM_NODEID:-0}" == "0" ]; then
    echo "Starting Ray head node on $(hostname)"
    
    # Build ray start command with optional GPU specification
    RAY_CMD="ray start --head --port=$RAY_NODE_PORT --dashboard-host=0.0.0.0"
    if [ -n "$GPUS_PER_NODE" ] && [ "$GPUS_PER_NODE" != "auto" ]; then
        RAY_CMD="$RAY_CMD --num-gpus=$GPUS_PER_NODE"
    fi
    
    $RAY_CMD
    echo "Ray head started on port $RAY_NODE_PORT"
    
    # Export for downstream use
    export RAY_ADDRESS="127.0.0.1:$RAY_NODE_PORT"
else
    echo "Starting Ray worker node $SLURM_NODEID on $(hostname)"
    
    # Wait for head to be ready
    echo "Waiting for Ray head at ${MASTER_ADDR}:${RAY_NODE_PORT}..."
    sleep 10
    
    export RAY_ADDRESS="${MASTER_ADDR}:${RAY_NODE_PORT}"
    
    # Build ray start command with optional GPU specification
    RAY_CMD="ray start --address=${RAY_ADDRESS}"
    if [ -n "$GPUS_PER_NODE" ] && [ "$GPUS_PER_NODE" != "auto" ]; then
        RAY_CMD="$RAY_CMD --num-gpus=$GPUS_PER_NODE"
    fi
    
    # Start worker without --block so we can control lifecycle
    $RAY_CMD
    
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
