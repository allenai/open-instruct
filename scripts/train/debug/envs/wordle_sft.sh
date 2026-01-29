#!/bin/bash
# Single GPU debug: Wordle env with SFT model that can actually play
#
# Uses PrimeIntellect/Qwen3-1.7B-Wordle-SFT - a model fine-tuned on Wordle
# This should get non-zero rewards since the model knows how to play

# Install Prime Intellect Wordle environment
uv run --extra prime-intellect prime env install will/wordle 2>/dev/null || true

VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run --extra prime-intellect open_instruct/grpo_fast.py \
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
    --model_name_or_path PrimeIntellect/Qwen3-1.7B-Wordle-SFT \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --learning_rate 3e-7 \
    --total_episodes 64 \
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
