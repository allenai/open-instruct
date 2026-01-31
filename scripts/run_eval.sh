#!/bin/bash
# Evaluate inference results
#
# Usage:
#   ./scripts/run_eval.sh <results_dir> [output_file]
#
# Example:
#   ./scripts/run_eval.sh /path/to/results
#   ./scripts/run_eval.sh results/ eval_output.json

set -e

RESULTS_DIR=${1:?Usage: ./scripts/run_eval.sh <results_dir> [output_file]}
OUTPUT_FILE=${2:-""}

echo "=== Evaluating Results ==="
echo "Results directory: $RESULTS_DIR"
echo ""

# Build command
CMD="uv run python scripts/eval_results.py --results_dir $RESULTS_DIR"

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output_file $OUTPUT_FILE"
fi

# Run evaluation
$CMD
