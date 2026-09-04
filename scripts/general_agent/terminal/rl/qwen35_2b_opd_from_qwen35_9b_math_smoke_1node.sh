#!/bin/bash

# One-node math-only OPD smoke test: Qwen3.5-2B student pure-distills from
# Qwen3.5-9B on DAPO math prompts. The student generates each completion; the
# frozen teacher scores those same tokens. The math verifier is retained for
# correctness diagnostics, but --opd_pure means verifier rewards carry no
# gradient.
#
# This is intentionally a plumbing test rather than a training run: 64
# rollouts/update and 128 total episodes give two optimizer updates. There are
# no tools, environment actors, teacher routing, or local benchmark evals.
# Layout: four ZeRO-3 learner ranks (including the sharded 9B teacher) and four
# one-GPU vLLM rollout engines on a single 8-GPU node.
set -euo pipefail

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"
shift

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
TOKENIZER="${TOKENIZER:-$MODEL}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-9B}"
EXP_NAME="${EXP_NAME:-qwen35_2b_opd_from_qwen35_9b_math_smoke_1node}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
PRIORITY="${PRIORITY:-high}"

uv run python mason.py \
    --task_name "$EXP_NAME" \
    --description "$RUN_NAME: pure OPD Qwen3.5-2B student from Qwen3.5-9B teacher on filtered DAPO math" \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority "$PRIORITY" \
    --pure_docker_mode \
    --image "$BEAKER_IMAGE" \
    --preemptible \
    --num_nodes 1 \
    --max_retries 1 \
    --timeout 2h \
    --gpus 8 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    --env VLLM_DISABLE_COMPILE_CACHE=1 \
    --env VLLM_USE_V1=1 \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --no_auto_dataset_cache \
    -- \
source configs/beaker_configs/ray_node_setup.sh \
\&\& uv run open_instruct/grpo_fast.py \
    --run_name "$RUN_NAME" \
    --exp_name "$EXP_NAME" \
    --model_name_or_path "$MODEL" \
    --tokenizer_name_or_path "$TOKENIZER" \
    --dataset_mixer_list hamishivi/DAPO-Math-17k-Processed_filtered 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 4 \
    --inflight_updates true \
    --opd_teacher_model_name_or_path "$TEACHER_MODEL" \
    --opd_kl_coef 1.0 \
    --opd_pure \
    --filter_zero_std_samples false \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --total_episodes 128 \
    --num_epochs 1 \
    --deepspeed_stage 3 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_enable_prefix_caching \
    --vllm_gdn_prefill_backend triton \
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
    --chat_template qwen_instruct_user_boxed_math \
    --mask_truncated_completions false \
    --gradient_checkpointing \
    --local_eval_every -1 \
    --save_freq -1 \
    --checkpoint_state_freq -1 \
    --save_traces \
    --save_trainer_logprobs false \
    --rollouts_save_path /weka/oe-adapt-default/allennlp/deletable_rollouts/ \
    --with_tracking \
    --seed 42 \
    --push_to_hub false \
    --try_auto_save_to_beaker false "$@"
