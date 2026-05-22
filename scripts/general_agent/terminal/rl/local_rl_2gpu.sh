#!/bin/bash

# Local 2-GPU debug run for terminal RL (GRPO + swerl_sandbox).
# Layout: 1 learner GPU + 1 vLLM engine GPU.
# Uses Qwen3-0.6B + small dataset slice. Docker must be running locally.
# No Podman services, no Beaker, no mason.py.

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export SWERL_DOCKER_AUTO_REMOVE=1
export SWERL_SANDBOX_TIMING_LOGS=1

uv run python open_instruct/grpo_fast.py \
    --exp_name tmax_rl_local_2gpu \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_mixer_list hamishivi/swerl-tmax-10k 32 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 1024 \
    --response_length 4096 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --total_episodes 32 \
    --deepspeed_stage 2 \
    --num_learners_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_enforce_eager \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --advantage_normalization_type centered \
    --verification_reward 1.0 \
    --temperature 1.0 \
    --tools swerl_sandbox \
    --tool_configs '{"task_data_hf_repo": "hamishivi/swerl-tmax-10k", "test_timeout": 60, "image": "python:3.12-slim"}' \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_sandbox_system_prompt.txt \
    --pool_size 16 \
    --max_steps 10 \
    --backend_timeout 300 \
    --gradient_checkpointing \
    --save_traces \
    --local_eval_every 8 \
    --logging_steps 1 \
    --seed 42 \
    --report_to none \
    --with_tracking \
    --output_dir output/tmax_rl_local_2gpu \
    --push_to_hub false
