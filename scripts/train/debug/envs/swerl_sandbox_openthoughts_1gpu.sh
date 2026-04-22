#!/bin/bash
# 1-GPU local debug variant of
# scripts/tmax/qwen_35_base_500step/qwen35_9b_base_agent_task_openthoughts.sh
#
# - Scales the 4-node / 32-GPU run down to a single GPU on the local machine.
# - Swaps Qwen/Qwen3.5-9B for Qwen/Qwen3.5-0.8B.
# - Drops mason / beaker wrapping; runs grpo_fast.py directly.
# - Preserves the openthoughts-specific training config (dataset, verification
#   reward, active sampling, inflight updates, advantage normalization, system
#   prompt) so behavior matches the production run as closely as possible.
#
# Requirements:
# - Apptainer >= 1.1 on PATH (uses python:3.12-slim task image by default).
#   The plain tag is accepted; ApptainerBackend prepends docker:// internally.
# - 1 GPU available.
# - Optional: set APPTAINER_CACHEDIR / APPTAINER_TMPDIR to fast scratch
#   to avoid filling $HOME with converted SIFs.

set -e

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

echo "Starting SWERL Sandbox openthoughts GRPO (1 GPU, Qwen3.5-0.8B)..."

uv run python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/agent-task-openthoughts 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3.5-0.8B \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 64 \
    --lr_scheduler_type constant \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --seed 42 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --single_gpu_mode \
    --push_to_hub false \
    --save_traces \
    --tools swerl_sandbox \
    --tool_configs '{"backend": "apptainer", "task_data_hf_repo": "hamishivi/agent-task-openthoughts", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 8 \
    --max_steps 10 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_sandbox_system_prompt.txt \
    --active_sampling \
    --backend_timeout 1200 \
    --inflight_updates true \
    --advantage_normalization_type centered \
    --local_eval_every 10 \
    --save_freq 20 \
    --output_dir output/swerl_sandbox_openthoughts_1gpu_debug

echo "Training complete!"
