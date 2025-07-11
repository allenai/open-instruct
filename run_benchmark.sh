#!/bin/bash

uv run python -m open_instruct.benchmark_generators \
    --model_name_or_path "hamishivi/qwen2_5_openthoughts2" \
    --tokenizer_name_or_path "hamishivi/qwen2_5_openthoughts2" \
    --dataset_mixer_list "hamishivi/hamishivi_rlvr_orz_math_57k_collected_all_filtered_hamishivi_qwen2_5_openthoughts2" "1.0" \
    --dataset_mixer_list_splits "train" \
    --max_token_length 10240 \
    --max_prompt_token_length 2048 \
    --temperature 1.0 \
    --response_length 64 \
    --vllm_top_p 0.9 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 16 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.9 \
    --pack_length 20480 \
    --chat_template_name "tulu_thinker" \
    --trust_remote_code \
    --seed 42 \
    --dataset_local_cache_dir "benchmark_cache" \
    --dataset_cache_mode "local" \
    --dataset_transform_fn "rlvr_tokenize_v1" "rlvr_filter_v1" \
    "$@"
