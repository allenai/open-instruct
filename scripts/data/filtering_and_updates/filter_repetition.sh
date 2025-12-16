#!/bin/bash

# Script to filter repetitive content from datasets
# Usage: ./scripts/data/filtering_and_updates/filter_repetition.sh --input-dataset <dataset_name> [options]

set -e

# Default values
INPUT_DATASET=""
COLUMN="messages"
SPLIT="train"
NUM_PROC=16
FILTER_USER_TURNS=""
DEBUG=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --input-dataset)
      INPUT_DATASET="$2"
      shift 2
      ;;
    --column)
      COLUMN="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --num-proc)
      NUM_PROC="$2"
      shift 2
      ;;
    --filter-user-turns)
      FILTER_USER_TURNS="--filter-user-turns"
      shift
      ;;
    --debug)
      DEBUG="--debug"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 --input-dataset <dataset_name> [options]"
      echo ""
      echo "Options:"
      echo "  --input-dataset <name>    Required: Input dataset name"
      echo "  --column <name>           Optional: Column name containing messages (default: messages)"
      echo "  --split <name>            Optional: Dataset split to process (default: train)"
      echo "  --num-proc <n>            Optional: Number of processes (default: 16)"
      echo "  --filter-user-turns       Optional: Also filter user messages"
      echo "  --debug                   Optional: Debug mode - only load first 1000 examples"
      echo "  -h, --help                Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check required arguments
if [[ -z "$INPUT_DATASET" ]]; then
  echo "Error: --input-dataset is required"
  echo "Use --help for usage information"
  exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Repetition Filtering"
echo "=========================================="
echo "Input dataset: $INPUT_DATASET"
echo "Column: $COLUMN"
echo "Split: $SPLIT"
echo "Filter user turns: $(if [[ -n "$FILTER_USER_TURNS" ]]; then echo "Yes"; else echo "No"; fi)"
echo "Debug mode: $(if [[ -n "$DEBUG" ]]; then echo "Yes (1000 samples)"; else echo "No"; fi)"
echo ""

# Run the filtering script
echo "Running repetition filtering..."
echo "Command: uv run python $SCRIPT_DIR/filter_ngram_repetitions.py --input-dataset $INPUT_DATASET --column $COLUMN --split $SPLIT --num-proc $NUM_PROC --verbose --push-to-hf $FILTER_USER_TURNS $DEBUG --manual-filter"
echo ""

if uv run python "$SCRIPT_DIR/filter_ngram_repetitions.py" \
    --input-dataset "$INPUT_DATASET" \
    --column "$COLUMN" \
    --split "$SPLIT" \
    --num-proc "$NUM_PROC" \
    --verbose \
    --push-to-hf \
    $FILTER_USER_TURNS \
    $DEBUG \
    --manual-filter; then

    echo "=========================================="
    echo "Repetition filtering completed successfully!"
    echo "=========================================="
else
    echo "=========================================="
    echo "ERROR: Repetition filtering failed!"
    echo "=========================================="
    exit 1
fi
