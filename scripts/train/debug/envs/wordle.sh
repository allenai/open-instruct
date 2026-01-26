#!/bin/bash
# Single GPU debug: Wordle env + python tool
#
# Prerequisites:
#   1. Install Prime Intellect dependencies:
#      uv pip install .[prime-intellect]
#   2. Install the Wordle environment:
#      prime env install will/wordle
#
# Dataset: hamishivi/wordle_env_train
#   - Uses verifier_source: "env" for EnvVerifier (reward from env.step())
#   - env_info.env_config passed to PrimeIntellectEnv

VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/wordle_env_train 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/wordle_env_train 4 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 1024 \
    --response_length 512 \
    --pack_length 2048 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --learning_rate 3e-7 \
    --total_episodes 100 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 3 \
    --local_eval_every 1 \
    --vllm_sync_backend gloo \
    --single_gpu_mode \
    --vllm_gpu_memory_utilization 0.3 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --tools wordle python \
    --tool_call_names wordle code \
    --tool_configs '{"env_name": "will/wordle"}' '{"api_endpoint": "https://open-instruct-tool-server.run.app/execute", "timeout": 3}' \
    --tool_parser_type vllm_hermes \
    --max_tool_calls 10 \
    --push_to_hub false
