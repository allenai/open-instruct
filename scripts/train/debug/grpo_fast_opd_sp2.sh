#!/bin/bash

# Local 4-GPU SP=2 repro for the mixed-geometry OPD teacher path:
# Qwen3.5-0.8B student (8 q-heads) + allenai/tmax-4b teacher (16 q-heads)
# forces full-sequence teacher scoring under Ulysses SP — the exact code path
# that failed on Beaker with the 27B teacher. 2 learners (SP=2) + 2 engines.
set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

TEACHER_MODEL=${TEACHER_MODEL:-allenai/tmax-4b}

uv run --active open_instruct/grpo_fast.py \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3.5-0.8B \
    --opd_teacher_model_name_or_path "$TEACHER_MODEL" \
    --opd_kl_coef 1.0 \
    --opd_pure \
    --filter_zero_std_samples false \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 96 \
    --deepspeed_stage 3 \
    --sequence_parallel_size 2 \
    --num_epochs 1 \
    --num_learners_per_node 2 \
    --vllm_num_engines 2 \
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
    --lm_head_fp32 true \
    --advantage_normalization_type centered \
    --seed 3 \
    --local_eval_every -1 \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --push_to_hub false "$@"
