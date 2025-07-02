#!/bin/bash
set -e
output_dir="output"

# Store raw patterns in an array. Single quotes prevent early brace expansion.
patterns=(
  '1751082866_saurabh5_rlvr-code-data-rust_{1,2}'
  '1751082866_saurabh5_rlvr-code-data-swift_{0,1,2}'
  '1751082866_saurabh5_rlvr-code-data-kotlin_{0,1,2}'
  '1751082866_saurabh5_rlvr-code-data-haskell_{0,1,2}'
  '1751082866_saurabh5_rlvr-code-data-lean_{0,1,2}'
  '1751082866_saurabh5_rlvr-code-data-typescript_{0,1,2}'
)

for batch_file_pattern in "${patterns[@]}"; do
    echo "Retrying pattern: $batch_file_pattern"

    # Expand the pattern at runtime and convert to comma-separated list
    expanded_files=$(eval echo "$output_dir/batch_files/${batch_file_pattern}.jsonl")
    comma_separated_paths=$(echo "$expanded_files" | tr ' ' ',')
    echo "Comma-separated paths: $comma_separated_paths"

    # Uncomment to run
    python retry_batch_files.py --batch_file_paths "$comma_separated_paths" --output_dir "$output_dir"
done