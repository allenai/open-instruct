#!/usr/bin/env bash
set -euo pipefail

# Usage: hf_copy_revision_to_main.sh <hf_repo_id> <source_revision>

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <hf_repo_id> <source_revision>" >&2
  exit 1
fi

HF_REPO_ID="$1"
SOURCE_REVISION="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PYTHON_SCRIPT="$PROJECT_ROOT/scripts/hf_copy_revision_to_main.py"

CMD=("python" "$PYTHON_SCRIPT" "$HF_REPO_ID" "$SOURCE_REVISION")

echo "Running: ${CMD[*]}"
"${CMD[@]}"
