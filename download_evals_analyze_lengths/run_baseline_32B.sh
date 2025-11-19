#!/bin/bash

# Array of model configurations
# Each entry: "alias|model_name1,model_name2,..."
configs=(
    "Qwen3-32B-no-thinking|Qwen3-32B-no-thinking"
    "hf-Qwen-Qwen2.5-32B-Instruct|hf-Qwen-Qwen2.5-32B-Instruct"
    "hf-Qwen-Qwen3-VL-32B-Thinking|hf-Qwen-Qwen3-VL-32B-Thinking"
    "hf-Qwen-Qwen3-VL-32B-Instruct|hf-Qwen-Qwen3-VL-32B-Instruct"
)

# Refresh DuckDB cache once first to avoid lock conflicts
echo "Refreshing DuckDB cache once before parallel execution..."
IFS='|' read -r alias models <<< "${configs[0]}"
IFS=',' read -ra model_array <<< "$models"
python download_evals_analyze_lengths/download_and_analyze.py "${model_array[@]}" \
    --model-alias "$alias" \
    --output-dir /weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/data/32B/"$alias"

# Now run the rest in parallel (they'll use the fresh cache)
for i in "${!configs[@]}"; do
    if [ $i -eq 0 ]; then
        continue  # Skip first one, already done above
    fi
    IFS='|' read -r alias models <<< "${configs[$i]}"
    IFS=',' read -ra model_array <<< "$models"
    python download_evals_analyze_lengths/download_and_analyze.py "${model_array[@]}" \
        --model-alias "$alias" \
        --output-dir /weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/data/32B/"$alias" \
        --no-refresh &
done

wait