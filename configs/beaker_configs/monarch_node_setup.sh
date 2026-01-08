#!/bin/bash
# Monarch node setup for Beaker multi-node training
#
# This script is similar to ray_node_setup.sh but sets up environment
# for Monarch-based distributed training instead of Ray.
#
# Usage: source configs/beaker_configs/monarch_node_setup.sh

export CURRENT_DATETIME=$(python -c "import datetime; import pytz; print(datetime.datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%m%d%y_%H%M%S'))")
export PYTHONPATH=$REPO_PATH
export PATH="/root/.local/bin:$PATH"

# NCCL performance tuning
export NCCL_CUMEM_ENABLE=0

echo "CURRENT_DATETIME=$CURRENT_DATETIME"
echo "PYTHONPATH=$PYTHONPATH"
echo "PATH=$PATH"

# Get leader IP from hostname
BEAKER_LEADER_REPLICA_IP=$(getent hosts ${BEAKER_LEADER_REPLICA_HOSTNAME} | awk '{print $1}')

# Monarch rendezvous port
MONARCH_RENDEZVOUS_PORT=9999

# Create Triton autotune cache directory
mkdir -p "$HOME/.triton/autotune"

# Export Monarch-specific environment variables
export MONARCH_LEADER_IP=$BEAKER_LEADER_REPLICA_IP
export MONARCH_REPLICA_RANK=$BEAKER_REPLICA_RANK
export MONARCH_RENDEZVOUS_PORT=$MONARCH_RENDEZVOUS_PORT

echo "Monarch environment setup:"
echo "  MONARCH_LEADER_IP=$MONARCH_LEADER_IP"
echo "  MONARCH_REPLICA_RANK=$MONARCH_REPLICA_RANK"
echo "  MONARCH_RENDEZVOUS_PORT=$MONARCH_RENDEZVOUS_PORT"

# Note: Unlike Ray which starts a daemon process, Monarch's process mesh
# is created within the Python application. This script just sets up
# the environment variables needed for the Python rendezvous.

if [ "$BEAKER_REPLICA_RANK" == "0" ]; then
    echo "This is the leader replica (rank 0)"
    echo "The leader will coordinate rendezvous for all workers"
else
    echo "This is worker replica (rank $BEAKER_REPLICA_RANK)"
    echo "Will connect to leader at $MONARCH_LEADER_IP:$MONARCH_RENDEZVOUS_PORT"
fi
