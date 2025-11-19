#!/bin/bash

# Array of model configurations
# Each entry: "alias|model_name1,model_name2,..."
configs=(
    "Qwen3-8B-Instruct|hf-qwen3-8b-instruct-no-thinking"
    "Qwen2.5-7B-Instruct|hf-Qwen-Qwen2.5-7B-Instruct-run1"
    "Qwen3-VL-8B-Instruct|hf-Qwen-Qwen3-VL-8B-Instruct-run3"
    "OLMo-2-B-Instruct|hf-allenai-OLMo-2-1124-7B-Instruct"
    "Olmo3-7B-Instruct|grpo_all_mix_p64_4_8k_olmo3-r-nonreason-sft-lc-permissive-dpo__1__1760977564_step_550_2"
)

# Refresh DuckDB cache once first to avoid lock conflicts
echo "Refreshing DuckDB cache once before parallel execution..."
IFS='|' read -r alias models <<< "${configs[0]}"
IFS=',' read -ra model_array <<< "$models"
python download_evals_analyze_lengths/download_and_analyze.py "${model_array[@]}" \
    --model-alias "$alias" \
    --output-dir /weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/data/7B/"$alias"

# Now run the rest in parallel (they'll use the fresh cache)
for i in "${!configs[@]}"; do
    if [ $i -eq 0 ]; then
        continue  # Skip first one, already done above
    fi
    IFS='|' read -r alias models <<< "${configs[$i]}"
    IFS=',' read -ra model_array <<< "$models"
    python download_evals_analyze_lengths/download_and_analyze.py "${model_array[@]}" \
        --model-alias "$alias" \
        --output-dir /weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/data/7B/"$alias" \
        --no-refresh &
done

wait


python download_evals_analyze_lengths/extract_medians_from_statistics.py /weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/data/7B

python download_evals_analyze_lengths/make_length_pareto_plot.py --output-dir /weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/plots/7B