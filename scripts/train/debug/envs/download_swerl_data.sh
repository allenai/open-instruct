#!/bin/bash
# Download SWERL task data from HuggingFace and export TASK_DATA_DIR.
# Source this script (or run it) before training.
set -euo pipefail

echo "Downloading task data from HuggingFace..."
REPO_DIR=$(uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('hamishivi/agent-task-combined', repo_type='dataset'))")

# Extract tar.gz if task directories don't already exist
if [ ! -d "$REPO_DIR/task_0" ]; then
    echo "Extracting task-data.tar.gz..."
    tar -xzf "$REPO_DIR/task-data.tar.gz" -C "$REPO_DIR"
fi

export TASK_DATA_DIR="$REPO_DIR"
echo "Task data at: $TASK_DATA_DIR"
