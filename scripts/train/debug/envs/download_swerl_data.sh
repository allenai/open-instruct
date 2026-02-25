#!/bin/bash
# Download SWERL task data from HuggingFace and export TASK_DATA_DIR.
# Source this script (or run it) before training.
set -euo pipefail

echo "Downloading task data from HuggingFace..."
TASK_DATA_DIR=$(uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('hamishivi/agent-task-combined', repo_type='dataset'))")
export TASK_DATA_DIR
echo "Task data at: $TASK_DATA_DIR"
