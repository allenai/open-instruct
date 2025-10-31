search_strings=(
    "hf-Qwen3-8B-3"
    "hf-OpenThinker3-7B-3"
    "hf-DeepSeek-R1-Distill-Qwen-7B-3"
    "hf-NVIDIA-Nemotron-Nano-9B-v2-3"
    "Qwen3-8B-no-thinking"
    #"Qwen3-32B-no-thinking"
    #"hf-Qwen-Qwen2.5-32B-Instruct"
    "hf-Qwen-Qwen3-VL-8B-Thinking"
    #"hf-Qwen-Qwen3-VL-32B-Thinking"
    "hf-Qwen-Qwen3-VL-8B-Instruct"
    #"hf-Qwen-Qwen3-VL-32B-Instruct"
    "olmo3_dpo_rl_final_mix_jupiter_10108_6245__1__1759426652_step_1400" #olmo 2 7B thinker
    "grpo_all_mix_p64_4_8k_olmo3-r-nonreason-sft-lc-permissive-dpo__1__1760977564_step_550_2" #olmo 3 7B instruct
)

for search_string in "${search_strings[@]}"; do
    python download_evals_analyze_lengths/download_and_analyze.py  $search_string --output-dir length_analyses/baseline/$search_string&
done

wait