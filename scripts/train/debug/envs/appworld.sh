#!/bin/bash
# Single GPU debug: AppWorld env + python tool
#
# Prerequisites:
#   Docker must be running (containers started automatically by setup_fn)
#
# Note: Uses appworld from git (dev version with pydantic v2 support)
#
# Dataset: hamishivi/appworld_env_train
#   - Uses verifier_source: "env" for EnvVerifier (reward from env.step())
#   - env_info.env_config.task_id specifies which AppWorld task to load

# Set AppWorld root to project directory (so Ray workers can find data)
export APPWORLD_ROOT="$(pwd)"

VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run --extra appworld open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/appworld_env_train_fixed 32 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/appworld_env_train_fixed 4 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 4096 \
    --response_length 2048 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 2 \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --learning_rate 3e-7 \
    --total_episodes 80 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 3 \
    --local_eval_every 1 \
    --vllm_sync_backend gloo \
    --single_gpu_mode \
    --vllm_gpu_memory_utilization 0.5 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --tools appworld python \
    --tool_call_names appworld code \
    --tool_configs '{"pool_size": 4}' '{"api_endpoint": "https://open-instruct-tool-server.run.app/execute", "timeout": 3}' \
    --tool_parser_type vllm_hermes \
    --max_tool_calls 25 \
    --push_to_hub false
