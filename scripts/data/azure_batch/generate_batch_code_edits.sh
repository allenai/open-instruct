#!/bin/bash
set -e

# Define datasets
#languages=("JavaScript" "bash" "cpp" "Go" "Java" "Rust" "Swift" "Kotlin" "Haskell" "Lean" "TypeScript" "Python")
# Removing the languages that have already been submitted. 
languages=("Python")
datasets=()
for language in "${languages[@]}"; do
    dataset="saurabh5/rlvr-code-data-$language"
    datasets+=("$dataset")
done
echo "Processing datasets: ${datasets[@]}"
datasets_str=$(IFS=','; echo "${datasets[*]}")
python batch_code_edit.py \
    --datasets "$datasets_str" \
    --num-errors 3 \
    --split "train" \
    --model "o3" \
    --output-dir "output"