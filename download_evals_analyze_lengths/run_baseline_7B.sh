#!/bin/bash

search_strings=(
    "hf-qwen3-8b-instruct-no-thinking"
    "hf-Qwen3-8B-3"
    "open_thoughts_OpenThinker3_7B_thinking_baselines_new_r3"
    "deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_thinking_baselines_new_r3"
    "nvidia_NVIDIA_Nemotron_Nano_9B_v2_thinking_baselines_new_r3"
    "Qwen_Qwen3_VL_8B_Thinking_thinking_baselines_new_r3"
    #"hf-Qwen-Qwen3-VL-8B-Instruct-3"
    "hf-Qwen-Qwen3-VL-8B-Instruct-run3",
    "allenai_Olmo_3_Thinking_RL_thinking_baselines_new_r3" #olmo 2 7B thinker
    "grpo_all_mix_p64_4_8k_olmo3-r-nonreason-sft-lc-permissive-dpo__1__1760977564_step_550_2" #olmo 3 7B instruct
)

model_names=(
    "Qwen3-8B-Instruct"
    "Qwen3-8B-Thinking"
    "OpenThinker3-7B"
    "DeepSeek-R1-Distill-Qwen-7B"
    "Nemotron-Nano-9B-v2"
    "Qwen3-VL-8B-Thinking"
    "Qwen3-VL-8B-Instruct"
    "Olmo3-7B-Thinking"
    "Olmo3-7B-Instruct"
)

# Refresh DuckDB cache once first to avoid lock conflicts
echo "Refreshing DuckDB cache once before parallel execution..."
python download_evals_analyze_lengths/download_and_analyze.py "${search_strings[0]}" \
    --model-alias "${model_names[0]}" \
    --output-dir /weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/data/7B/"${model_names[0]}"

# Now run the rest in parallel (they'll use the fresh cache)
for i in "${!search_strings[@]}"; do
    if [ $i -eq 0 ]; then
        continue  # Skip first one, already done above
    fi
    python download_evals_analyze_lengths/download_and_analyze.py "${search_strings[$i]}" \
        --model-alias "${model_names[$i]}" \
        --output-dir /weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/data/7B/"${model_names[$i]}" \
        --no-refresh &
done

wait


python download_evals_analyze_lengths/extract_medians_from_statistics.py /weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/data/7B

python download_evals_analyze_lengths/make_length_pareto_plot.py