#!/bin/bash

# Script to run Chinese character filtering on multiple datasets
# Usage: ./filter_chinese_batch.sh

# Set the threshold for Chinese character detection
THRESHOLD=0.05

# List of datasets to process
DATASETS=(
    "allenai/OpenThoughts3-merged-format-filtered-keyword-filtered-filter-datecutoff-ngram-filtered"
    "allenai/tulu_v3.9_wildchat_100k_english-r1-final-content-filtered"
    "allenai/wildchat-r1-p2-repetition-filter"
    "allenai/oasst1-r1-format-filtered-keyword-filtered-filter-datecutoff"
    "allenai/coconot-r1-format-domain-filtered-keyword-filtered-filter-datecutoff"
    "allenai/persona-precise-if-r1-final-content-filtered"
    "allenai/wildguardmix-r1-v2-all-filtered-ngram-filtered"
    "allenai/wildjailbreak-r1-v2-format-filtered-keyword-filtered-filter-datecutoff-final-content-filtered"
    "allenai/SYNTHETIC-2-SFT-format-filtered-keyword-filtered-filter-datecutoff-ngram-filtered"
    "allenai/aya-100k-r1-format-filtered-keyword-filtered-filter-datecutoff-ngram-filtered"
    "allenai/tablegpt_r1-format-filtered-keyword-filtered-filter-datecutoff"
    "allenai/the-algorithm-python-r1-format-filtered-keyword-filtered-filter-datecutoff"
    "allenai/acecoder-r1-format-filtered-keyword-filtered-filter-datecutoff-ngram-filtered"
    "allenai/rlvr-code-data-python-r1-format-filtered-keyword-filtered-filter-datecutoff-ngram-filtered"
    "allenai/sciriff_10k_r1-format-filtered-keyword-filtered-filter-datecutoff"
    "allenai/numinatmath-r1-format-filtered-keyword-filtered-filter-datecutoff"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Create logs directory
mkdir -p logs

# Start processing
print_status "Starting Chinese character filtering on ${#DATASETS[@]} datasets"
print_status "Threshold: $THRESHOLD"
echo

# Process each dataset
for dataset in "${DATASETS[@]}"; do
    print_status "Processing dataset: $dataset"

    # Create log file name
    log_file="logs/filter_chinese_$(echo $dataset | sed 's/[^a-zA-Z0-9]/_/g')_$(date +%Y%m%d_%H%M%S).log"

    # Run the filtering script
    print_status "Running: python scripts/data/filtering_and_updates/filter_chinese.py --input-dataset $dataset --threshold $THRESHOLD"

    if uv run python scripts/data/filtering_and_updates/filter_chinese.py --input-dataset "$dataset" --threshold $THRESHOLD > "$log_file" 2>&1; then
        print_success "Completed processing: $dataset"
        print_status "Log saved to: $log_file"
    else
        print_error "Failed to process: $dataset"
        print_status "Check log file for details: $log_file"
    fi

    echo
done

print_status "Batch processing completed!"
print_status "Check the logs directory for detailed output from each dataset"
