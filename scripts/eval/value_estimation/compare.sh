#!/bin/bash
# Aggregate score_dataset outputs into a consolidated markdown + csv table.
# Usage: compare.sh out_dir path1.parquet path2.parquet ...
set -euo pipefail

OUT_DIR="${1:?Output directory required}"
shift
PATHS=("$@")

mkdir -p "${OUT_DIR}"

uv run python -m open_instruct.value_estimation compare_runs \
    --score_dataset_paths "${PATHS[@]}" \
    --output_markdown_path "${OUT_DIR}/comparison.md" \
    --output_csv_path "${OUT_DIR}/comparison.csv"

echo "Wrote ${OUT_DIR}/comparison.md"
echo "Wrote ${OUT_DIR}/comparison.csv"
