#!/bin/bash

# Script to filter repetitive thinking traces from datasets
# Usage: ./scripts/data/filtering_and_updates/filter_thinking_repetitions.sh --input-dataset <dataset_name> [options]

set -e

# Default values
INPUT_DATASET=""
OUTPUT_DATASET=""
COLUMN="messages"
SPLIT="train"
NUM_PROC=16
MIN_STRATEGIES=2
DEBUG=""
VERBOSE=""
ANALYZE_ONLY=""
NO_POS_TAGGING=""
PUSH_TO_HF="--push-to-hf"
STREAMING=""
NUM_EXAMPLES=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --input-dataset)
      INPUT_DATASET="$2"
      shift 2
      ;;
    --output-dataset)
      OUTPUT_DATASET="$2"
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
    --min-strategies)
      MIN_STRATEGIES="$2"
      shift 2
      ;;
    --no-pos-tagging)
      NO_POS_TAGGING="--no-pos-tagging"
      shift
      ;;
    --debug)
      DEBUG="--debug"
      shift
      ;;
    --verbose)
      VERBOSE="--verbose"
      shift
      ;;
    --analyze-only)
      ANALYZE_ONLY="--analyze-only"
      shift
      ;;
    --no-push)
      PUSH_TO_HF=""
      shift
      ;;
    --streaming)
      STREAMING="--streaming"
      shift
      ;;
    --num-examples)
      NUM_EXAMPLES="--num-examples $2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 --input-dataset <dataset_name> [options]"
      echo ""
      echo "Options:"
      echo "  --input-dataset <name>    Required: Input dataset name"
      echo "  --output-dataset <name>   Optional: Output dataset name (default: <input>-thinking-filtered)"
      echo "  --column <name>           Optional: Column name containing messages (default: messages)"
      echo "  --split <name>            Optional: Dataset split to process (default: train)"
      echo "  --num-proc <n>            Optional: Number of processes (default: 16)"
      echo "  --min-strategies <n>      Optional: Min strategies to agree for flagging (default: 2)"
      echo "  --no-pos-tagging          Optional: Disable spaCy POS tagging (use word-class system)"
      echo "  --streaming               Optional: Stream dataset (avoids full download)"
      echo "  --num-examples <n>        Optional: Number of examples in streaming mode (default: 100)"
      echo "  --debug                   Optional: Debug mode - only process first 1000 examples"
      echo "  --verbose                 Optional: Print details of flagged examples"
      echo "  --analyze-only            Optional: Only analyze, don't push"
      echo "  --no-push                 Optional: Don't push to HuggingFace Hub"
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

# Build output dataset arg
OUTPUT_ARG=""
if [[ -n "$OUTPUT_DATASET" ]]; then
  OUTPUT_ARG="--output-dataset $OUTPUT_DATASET"
fi

echo "=========================================="
echo "Thinking Trace Repetition Filtering"
echo "=========================================="
echo "Input dataset: $INPUT_DATASET"
echo "Column: $COLUMN"
echo "Split: $SPLIT"
echo "Min strategies: $MIN_STRATEGIES"
echo "POS tagging: $(if [[ -n "$NO_POS_TAGGING" ]]; then echo "Disabled (word-class)"; else echo "Enabled (spaCy)"; fi)"
echo "Streaming: $(if [[ -n "$STREAMING" ]]; then echo "Yes"; else echo "No"; fi)"
echo "Debug mode: $(if [[ -n "$DEBUG" ]]; then echo "Yes (1000 samples)"; else echo "No"; fi)"
echo "Analyze only: $(if [[ -n "$ANALYZE_ONLY" ]]; then echo "Yes"; else echo "No"; fi)"
echo ""

# Run the filtering script
# Use --extra filtering to ensure spaCy is available for POS tagging
echo "Running thinking trace repetition filtering..."
UV_RUN="uv run --extra filtering"
if [[ -n "$NO_POS_TAGGING" ]]; then
  # No need for spaCy if POS tagging is disabled
  UV_RUN="uv run"
fi
CMD="$UV_RUN python $SCRIPT_DIR/filter_thinking_repetitions.py --input-dataset $INPUT_DATASET --column $COLUMN --split $SPLIT --num-proc $NUM_PROC --min-strategies $MIN_STRATEGIES $OUTPUT_ARG $NO_POS_TAGGING $DEBUG $VERBOSE $ANALYZE_ONLY $PUSH_TO_HF $STREAMING $NUM_EXAMPLES"
echo "Command: $CMD"
echo ""

if eval "$CMD"; then
    echo "=========================================="
    echo "Thinking trace repetition filtering completed successfully!"
    echo "=========================================="
else
    echo "=========================================="
    echo "ERROR: Thinking trace repetition filtering failed!"
    echo "=========================================="
    exit 1
fi
