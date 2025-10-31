#!/bin/bash

search_strings=(
    "hf-Qwen3-8B-3"
    "hf-OpenThinker3-7B-3"
    "hf-DeepSeek-R1-Distill-Qwen-7B-3"
    "hf-NVIDIA-Nemotron-Nano-9B-v2-3"
    "Qwen3-8B-no-thinking"
    "hf-Qwen-Qwen3-VL-8B-Thinking"
    "hf-Qwen-Qwen3-VL-8B-Instruct"
    "olmo3_dpo_rl_final_mix_jupiter_10108_6245__1__1759426652_step_1400" #olmo 2 7B thinker
    "grpo_all_mix_p64_4_8k_olmo3-r-nonreason-sft-lc-permissive-dpo__1__1760977564_step_550_2" #olmo 3 7B instruct
)

# Refresh DuckDB cache once first to avoid lock conflicts
echo "Refreshing DuckDB cache once before parallel execution..."
python download_evals_analyze_lengths/download_and_analyze.py "${search_strings[0]}" --output-dir /weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/data/7B/"${search_strings[0]}"

# Now run the rest in parallel (they'll use the fresh cache)
for search_string in "${search_strings[@]:1}"; do
    python download_evals_analyze_lengths/download_and_analyze.py  $search_string --output-dir /weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/data/7B/$search_string --no-refresh &
done

wait