#!/bin/bash

# Script to run keyword filtering followed by cutoff date filtering sequentially
# Usage: ./filter_datasets_sequential.sh --input-dataset <dataset_name> [--filter-user-turns] [--num-proc <num>] [--split <split>] [--column <column>]

set -e  # Exit on any error

# Default values
NUM_PROC=16
SPLIT="train"
COLUMN="messages"
FILTER_USER_TURNS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --input-dataset)
      INPUT_DATASET="$2"
      shift 2
      ;;
    --filter-user-turns)
      FILTER_USER_TURNS="--filter-user-turns"
      shift
      ;;
    --num-proc)
      NUM_PROC="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --column)
      COLUMN="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 --input-dataset <dataset_name> [options]"
      echo ""
      echo "Options:"
      echo "  --input-dataset <name>    Required: Input dataset name (e.g., allenai/tulu-3-sft-mixture)"
      echo "  --filter-user-turns       Optional: Also filter user messages (default: only assistant messages)"
      echo "  --num-proc <number>       Optional: Number of processes for parallel processing (default: 16)"
      echo "  --split <name>            Optional: Dataset split to process (default: train)"
      echo "  --column <name>           Optional: Column name containing messages (default: messages)"
      echo "  -h, --help                Show this help message"
      echo ""
      echo "This script runs two filtering steps sequentially:"
      echo "1. filter_dataset_by_keywords.py: Removes examples with provider-specific language"
      echo "2. filter_cutoff_date.py: Removes examples mentioning knowledge cutoff dates"
      echo ""
      echo "Output datasets:"
      echo "  Step 1: <input-dataset>-keyword-filtered"
      echo "  Step 2: <input-dataset>-keyword-filtered-filter-datecutoff"
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
echo "Sequential Dataset Filtering Pipeline"
echo "=========================================="
echo "Input dataset: $INPUT_DATASET"
echo "Filter user turns: $(if [[ -n "$FILTER_USER_TURNS" ]]; then echo "Yes"; else echo "No"; fi)"
echo "Number of processes: $NUM_PROC"
echo "Split: $SPLIT"
echo "Column: $COLUMN"
echo ""

# Step 1: Keyword filtering
echo "Step 1: Running keyword filtering..."
echo "Command: python $SCRIPT_DIR/filter_dataset_by_keywords.py --input-dataset $INPUT_DATASET $FILTER_USER_TURNS"
echo ""

python "$SCRIPT_DIR/filter_dataset_by_keywords.py" --input-dataset "$INPUT_DATASET" $FILTER_USER_TURNS

if [[ $? -ne 0 ]]; then
  echo "Error: Keyword filtering failed"
  exit 1
fi

# Determine intermediate dataset name (output from step 1)
if [[ "$INPUT_DATASET" == *"/"* ]]; then
  # Dataset has organization prefix
  INTERMEDIATE_DATASET="${INPUT_DATASET}-keyword-filtered"
else
  # No organization prefix
  INTERMEDIATE_DATASET="${INPUT_DATASET}-keyword-filtered"
fi

echo ""
echo "Step 1 completed. Intermediate dataset: $INTERMEDIATE_DATASET"
echo ""

# Step 2: Cutoff date filtering
echo "Step 2: Running cutoff date filtering..."
echo "Command: python $SCRIPT_DIR/filter_cutoff_date.py --dataset $INTERMEDIATE_DATASET --column $COLUMN --num_proc $NUM_PROC --split $SPLIT --push_to_hf"
echo ""

python "$SCRIPT_DIR/filter_cutoff_date.py" --dataset "$INTERMEDIATE_DATASET" --column "$COLUMN" --num_proc "$NUM_PROC" --split "$SPLIT" --push_to_hf

if [[ $? -ne 0 ]]; then
  echo "Error: Cutoff date filtering failed"
  exit 1
fi

# Determine final dataset name
FINAL_DATASET="${INTERMEDIATE_DATASET}-filter-datecutoff"

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo "Original dataset: $INPUT_DATASET"
echo "After keyword filtering: $INTERMEDIATE_DATASET"
echo "Final dataset: $FINAL_DATASET"
echo ""
echo "You can now use the final dataset: $FINAL_DATASET"
