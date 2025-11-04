#!/usr/bin/env bash
set -euo pipefail

# Usage: hf_upload_folder.sh <path> <hf_repo_id> <revision> [--private]

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <path> <hf_repo_id> <revision> [--private]" >&2
  exit 1
fi

PATH_TO_UPLOAD="$1"
HF_REPO_ID="$2"
REVISION="$3"
PRIVATE_FLAG="${4:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PYTHON_SCRIPT="$PROJECT_ROOT/scripts/hf_upload_folder.py"

if [[ ! -d "$PATH_TO_UPLOAD" ]]; then
  echo "Error: Path does not exist or is not a directory: $PATH_TO_UPLOAD" >&2
  exit 2
fi

CMD=("python" "$PYTHON_SCRIPT" "$PATH_TO_UPLOAD" "$HF_REPO_ID" "$REVISION")
if [[ "$PRIVATE_FLAG" == "--private" ]]; then
  CMD+=("--private")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"


