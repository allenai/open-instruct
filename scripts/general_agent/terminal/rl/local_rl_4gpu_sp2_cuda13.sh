#!/bin/bash

# CUDA-13 local 4-GPU debug run for terminal RL (GRPO + swerl_sandbox) with
# Ulysses SEQUENCE PARALLELISM (SP=2) — the faithful local test of the
# fa4-through-Ulysses-all-to-all path that the 2-GPU (SP=1) script can't exercise.
#
# Layout mirrors the holmes 4-GPU smoke: 2 learner GPUs (SP=2, stage 3) +
# 2 vLLM engine GPUs. Runs directly from the synced cu13 env (deepspeed 0.19.3
# with the Ulysses attn *blocklist* → fa4 permitted). No Beaker, no image rebuild.
#
# Keeps the small 0.6B model + swerl_sandbox dataset because that combo produces
# real reward variance locally (unlike vanillux terminal tasks), so it reaches a
# REAL gradient step → actually pushes tokens through fa4's flash_attn.cute
# interface inside Ulysses' all-to-all resharding ("permitted != validated").
#
# Requires 4 visible GPUs + local Docker. Source env.cuda13.sh first.

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export SWERL_DOCKER_AUTO_REMOVE=1
export SWERL_SANDBOX_TIMING_LOGS=1

uv run --no-default-groups --group dev --group cuda13 python open_instruct/grpo_fast.py \
    --exp_name terminal_local_rl_tmax_4gpu_sp2_cuda13 \
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
    --deepspeed_stage 3 \
    --sequence_parallel_size 2 \
    --attn_implementation flash_4 \
    --num_learners_per_node 2 \
    --vllm_num_engines 2 \
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
    --report_to wandb \
    --with_tracking \
    --wandb_project oe-general-agents \
    --output_dir output/tmax_rl_local_4gpu_sp2_cuda13 \
    --push_to_hub false
