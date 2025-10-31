#!/bin/bash

search_strings=(
    "Qwen3-32B-no-thinking"
    "hf-Qwen-Qwen2.5-32B-Instruct"
    "hf-Qwen-Qwen3-VL-32B-Thinking"
    "hf-Qwen-Qwen3-VL-32B-Instruct"
)

# Refresh DuckDB cache once first to avoid lock conflicts
echo "Refreshing DuckDB cache once before parallel execution..."
python download_evals_analyze_lengths/download_and_analyze.py "${search_strings[0]}" --output-dir /weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/data/32B/"${search_strings[0]}"

# Now run the rest in parallel (they'll use the fresh cache)
for search_string in "${search_strings[@]:1}"; do
    python download_evals_analyze_lengths/download_and_analyze.py  $search_string --output-dir /weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/data/32B/$search_string --no-refresh &
done

wait