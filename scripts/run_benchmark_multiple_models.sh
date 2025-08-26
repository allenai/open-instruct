#!/bin/bash
# Runs benchmarks for multiple models with a specified generation length
# Usage: ./run_benchmark_multiple_models.sh <generation_length> <model1> <model2> ...
# E.g. $ ./run_benchmark_multiple_models.sh 64000 hamishivi/qwen2_5_openthoughts2 another/model

set -e

# Check if at least 2 arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <generation_length> <model1> [model2] ..."
    echo "Example: $0 64000 hamishivi/qwen2_5_openthoughts2 another/model"
    exit 1
fi

# First argument is the generation length
generation_length="$1"
shift

# Validate that generation_length is a number
if ! [[ "$generation_length" =~ ^[0-9]+$ ]]; then
    echo "Error: Generation length must be a positive integer"
    exit 1
fi

echo "Running benchmarks with generation length: $generation_length"
echo "Models to benchmark: $@"
echo "----------------------------------------"

# Loop through remaining arguments (models)
for model in "$@"; do
    echo ""
    echo "Starting benchmark for model: $model"
    echo "Running: ./scripts/gantry_run_benchmark.sh $generation_length $model"
    
    # Call the gantry_run_benchmark.sh script with generation length and model
    ./scripts/gantry_run_benchmark.sh "$generation_length" "$model"
    
    echo "Completed benchmark for model: $model"
    echo "----------------------------------------"
done

echo ""
echo "All benchmarks completed successfully!"