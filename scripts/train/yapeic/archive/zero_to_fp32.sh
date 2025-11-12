#!/usr/bin/env bash

# Consolidate DeepSpeed ZeRO shards into a single fp32 checkpoint using zero_to_fp32.py
#
# Usage:
#   zero_to_fp32.sh \
#     --checkpoint_dir /path/to/checkpoints \
#     --tag global_step100 \
#     --output_file /tmp/consolidated_model.pt \
#     [--script_path /custom/path/to/zero_to_fp32.py] \
#     [--python python3]
#
# Notes:
# - Different DeepSpeed versions accept argument orders differently. This wrapper
#   tries both common forms automatically:
#     1) zero_to_fp32.py <ckpt_dir> --tag <tag> <output_file>
#     2) zero_to_fp32.py <ckpt_dir> <output_file> --tag <tag>

set -u -o pipefail

print_usage() {
  echo "Usage: $0 --checkpoint_dir DIR --tag TAG --output_file FILE [--script_path PATH] [--python PYTHON]" >&2
}

CHECKPOINT_DIR=""
TAG=""
OUTPUT_FILE=""
SCRIPT_PATH=""
PYTHON_BIN="python"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint_dir)
      CHECKPOINT_DIR="$2"; shift 2;;
    --tag)
      TAG="$2"; shift 2;;
    --output_file)
      OUTPUT_FILE="$2"; shift 2;;
    --script_path)
      SCRIPT_PATH="$2"; shift 2;;
    --python)
      PYTHON_BIN="$2"; shift 2;;
    -h|--help)
      print_usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage; exit 1;;
  esac
done

if [[ -z "$CHECKPOINT_DIR" || -z "$TAG" || -z "$OUTPUT_FILE" ]]; then
  echo "Missing required arguments." >&2
  print_usage
  exit 1
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "Checkpoint directory not found: $CHECKPOINT_DIR" >&2
  exit 1
fi

# Resolve zero_to_fp32.py path
if [[ -z "$SCRIPT_PATH" ]]; then
  if [[ -f "$CHECKPOINT_DIR/zero_to_fp32.py" ]]; then
    SCRIPT_PATH="$CHECKPOINT_DIR/zero_to_fp32.py"
  elif command -v zero_to_fp32.py >/dev/null 2>&1; then
    SCRIPT_PATH="$(command -v zero_to_fp32.py)"
  else
    echo "Could not locate zero_to_fp32.py. Provide --script_path explicitly." >&2
    exit 1
  fi
fi

echo "Using zero_to_fp32.py: $SCRIPT_PATH"
echo "Checkpoint dir       : $CHECKPOINT_DIR"
echo "Tag                  : $TAG"
echo "Output file          : $OUTPUT_FILE"

# Ensure output directory exists
OUT_DIR="$(dirname "$OUTPUT_FILE")"
mkdir -p "$OUT_DIR"

# Try common DeepSpeed invocations

set +e
"$PYTHON_BIN" "$SCRIPT_PATH" "$CHECKPOINT_DIR" --tag "$TAG" "$OUTPUT_FILE"
STATUS=$?
if [[ $STATUS -ne 0 ]]; then
  echo "First invocation form failed (status=$STATUS). Trying alternate form..." >&2
  "$PYTHON_BIN" "$SCRIPT_PATH" "$CHECKPOINT_DIR" "$OUTPUT_FILE" --tag "$TAG"
  STATUS=$?
fi
set -e

if [[ $STATUS -ne 0 ]]; then
  echo "zero_to_fp32.py failed with status $STATUS using both invocation forms." >&2
  exit $STATUS
fi

echo "Consolidation complete: $OUTPUT_FILE"

