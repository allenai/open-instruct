#!/bin/bash

# Script to test the n-gram repetition filtering script
# Note: This script tests the sentence-level repetition detection, not configurable n-gram parameters
# Usage: ./scripts/data/filtering_and_updates/test_ngram_removal_rates.sh --input-dataset <dataset_name> [options]

set -e

# Default values
INPUT_DATASET=""
COLUMN="messages"
SPLIT="train"
NUM_PROC=16
OUTPUT_DIR="ngram_test_results"
FILTER_USER_TURNS=""
DEBUG=""

# Since the current filter_ngram_repetitions.py doesn't support configurable parameters,
# we'll just test it once but keep the framework for future expansion
TEST_CONFIGS=(
    "default"
)

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
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --filter-user-turns)
      FILTER_USER_TURNS="--filter-user-turns"
      shift
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
      echo "  --output-dir <dir>        Optional: Output directory for results (default: ngram_test_results)"
      echo "  --filter-user-turns       Optional: Also filter user messages"
      echo "  --debug                   Optional: Debug mode - only load first 1000 examples for testing"
      echo "  -h, --help                Show this help message"
      echo ""
      echo "This script tests the n-gram repetition filtering script using sentence-level detection."
      echo "Currently it does not support configurable n-gram parameters."
      echo ""
      echo "Default test configurations:"
      for config in "${TEST_CONFIGS[@]}"; do
        echo "  $config (sentence-level repetition detection)"
      done
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

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "N-gram Repetition Filtering Test"
echo "=========================================="
echo "Input dataset: $INPUT_DATASET"
echo "Column: $COLUMN"
echo "Split: $SPLIT"
echo "Output directory: $OUTPUT_DIR"
echo "Filter user turns: $(if [[ -n "$FILTER_USER_TURNS" ]]; then echo "Yes"; else echo "No"; fi)"
echo "Debug mode: $(if [[ -n "$DEBUG" ]]; then echo "Yes (1000 samples)"; else echo "No"; fi)"
echo ""
echo "Note: Testing sentence-level repetition detection (current implementation)"
echo "Note: Only filtering assistant messages by default (not user messages)"
echo "Testing ${#TEST_CONFIGS[@]} configuration(s)..."
echo ""

# Initialize results file
RESULTS_FILE="$OUTPUT_DIR/removal_rates_summary.txt"
echo "N-gram Repetition Filtering Test Results" > "$RESULTS_FILE"
echo "Dataset: $INPUT_DATASET" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "Note: Using sentence-level repetition detection" >> "$RESULTS_FILE"
echo "=============================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
printf "%-15s %-15s %-15s %-15s\n" "Configuration" "Examples_Removed" "Removal_Rate%" "Output_File" >> "$RESULTS_FILE"
echo "------------------------------------------------------------" >> "$RESULTS_FILE"

# Test each configuration
for i in "${!TEST_CONFIGS[@]}"; do
    config="${TEST_CONFIGS[$i]}"
    
    echo "[$((i+1))/${#TEST_CONFIGS[@]}] Testing config: $config"
    
    # Generate output filename
    output_file="$OUTPUT_DIR/test_${config}.log"
    
    # Run the filtering script and capture output
    echo "Running: uv run python $SCRIPT_DIR/filter_ngram_repetitions.py --input-dataset $INPUT_DATASET --column $COLUMN --split $SPLIT --num-proc $NUM_PROC --verbose $FILTER_USER_TURNS $DEBUG"
    
    if uv run python "$SCRIPT_DIR/filter_ngram_repetitions.py" \
        --input-dataset "$INPUT_DATASET" \
        --column "$COLUMN" \
        --split "$SPLIT" \
        --num-proc "$NUM_PROC" \
        --verbose \
        $FILTER_USER_TURNS \
        $DEBUG > "$output_file" 2>&1; then
        
        # Extract statistics from output
        examples_removed=$(grep "Removed [0-9]* examples" "$output_file" | grep -o "Removed [0-9]*" | grep -o "[0-9]*" || echo "0")
        removal_rate=$(grep "Removed [0-9]* examples ([0-9]*\.[0-9]*%)" "$output_file" | grep -o "([0-9]*\.[0-9]*%)" | tr -d "()" || echo "0.00%")
        
        echo "  → Removed $examples_removed examples ($removal_rate)"
        
        # Log to results file
        printf "%-15s %-15s %-15s %-15s\n" "$config" "$examples_removed" "$removal_rate" "$(basename "$output_file")" >> "$RESULTS_FILE"
        
    else
        echo "  → ERROR: Failed to run configuration"
        printf "%-15s %-15s %-15s %-15s\n" "$config" "ERROR" "ERROR" "$(basename "$output_file")" >> "$RESULTS_FILE"
    fi
    
    echo ""
done

echo "" >> "$RESULTS_FILE"
echo "Detailed logs are available in the individual test files." >> "$RESULTS_FILE"

echo "=========================================="
echo "Testing completed!"
echo "=========================================="
echo "Results summary saved to: $RESULTS_FILE"
echo "Individual test logs saved to: $OUTPUT_DIR/"
echo ""
echo "Results summary:"
cat "$RESULTS_FILE"
