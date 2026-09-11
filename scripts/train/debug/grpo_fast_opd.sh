#!/bin/bash

# Local 2-GPU smoke test for on-policy distillation (OPD): Qwen3-0.6B student
# distills from a frozen Qwen3-1.7B teacher (same tokenizer) on gsm8k.
# GPU 0 hosts the learner (student + teacher eval copy), GPU 1 the vLLM engine.
#
# Mirrors the Terminal RL DPPO recipe (liger tiled loss + dppo + vllm logprobs +
# TIS cap 0) so the exact production loss path is exercised. --opd_pure makes
# the distillation term the only gradient signal; rewards are still computed
# and logged. filter_zero_std_samples is disabled so groups with identical
# rewards (useless for GRPO, fine for OPD) are kept.
set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
# When launched via `uv run`, ray re-wraps its workers in `uv run`, which can
# resolve to a cache-managed env and refuse to start. Harmless on Beaker.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

uv run --active open_instruct/grpo_fast.py \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --opd_teacher_model_name_or_path Qwen/Qwen3-1.7B \
    --opd_kl_coef 1.0 \
    --opd_pure \
    --filter_zero_std_samples false \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 96 \
    --deepspeed_stage 3 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --load_ref_policy false \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --loss_fn dppo \
    --dppo_divergence_type tv \
    --dppo_divergence_threshold 0.1 \
    --use_liger_grpo_loss \
    --liger_grpo_loss_chunk_size 8 \
    --advantage_normalization_type centered \
    --seed 3 \
    --local_eval_every -1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --push_to_hub false "$@"
