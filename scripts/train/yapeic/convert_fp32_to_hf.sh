#!/usr/bin/env bash

set -u -o pipefail

print_usage() {
  echo "Usage: $0 --fp32_path FILE --base_model MODEL --out_dir DIR [--strict] [--python PY]" >&2
}

FP32_PATH=""
BASE_MODEL=""
OUT_DIR=""
STRICT="false"
PYTHON_BIN="python"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fp32_path)
      FP32_PATH="$2"; shift 2;;
    --base_model)
      BASE_MODEL="$2"; shift 2;;
    --out_dir)
      OUT_DIR="$2"; shift 2;;
    --strict)
      STRICT="true"; shift 1;;
    --python)
      PYTHON_BIN="$2"; shift 2;;
    -h|--help)
      print_usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage; exit 1;;
  esac
done

if [[ -z "$FP32_PATH" || -z "$BASE_MODEL" || -z "$OUT_DIR" ]]; then
  echo "Missing required arguments." >&2
  print_usage
  exit 1
fi

if [[ ! -f "$FP32_PATH" ]]; then
  echo "fp32 file not found: $FP32_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CMD=("$PYTHON_BIN" "$SCRIPT_DIR/convert_fp32_to_hf.py" --fp32_path "$FP32_PATH" --base_model "$BASE_MODEL" --out_dir "$OUT_DIR")
if [[ "$STRICT" == "true" ]]; then
  CMD+=("--strict")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"


