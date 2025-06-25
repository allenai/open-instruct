#!/bin/bash
set -e

# Define datasets
datasets=(
    #"saurabh5/llama-nemotron-rlvr"
    #"saurabh5/the-algorithm-python"
    #"saurabh5/rlvr_acecoder"
    "saurabh5/open-code-reasoning-rlvr"
    "saurabh5/tulu-3-personas-code-rlvr"
)

# Join datasets with commas
dataset_string=$(IFS=','; echo "${datasets[*]}")

echo "Processing datasets: $dataset_string"
python batch_code_edit.py \
    --datasets "$dataset_string" \
    --num-errors 3 \
    --split "train" \
    --sample-limit 10 \
    --model "o3" \
    --output-dir "output"