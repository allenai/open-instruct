#!/bin/bash
# THUNLP "Rethinking OPD" main math recipe (Section 3.1 / Appendix A.2), adapted for
# open-instruct grpo_fast on Beaker.
#
# Paper targets:
#   Student:  Qwen/Qwen3-1.7B-Base
#   Teacher:  lllyx/Qwen3-4B-Base-GRPO  (GRPO on DAPO-Math-17K; see their Appendix A.1)
#   Data:     DAPO-Math-17K, 1 epoch, non-thinking chat template
#   Train:    global batch 64 prompts, 4 rollouts/prompt, lr 1e-6, T=1.0, top_p=1.0
#             max prompt 1024, max response 7168
#   Eval:     AIME24/25 + AMC, avg@16, T=0.7, top_p=0.95, max response 31744
#             (use THUNLP scripts/val/ — do not trust in-loop verl val on v0.7.0)
#
# IMPORTANT — not bit-identical to thunlp/OPD verl_example/opd.sh:
#   1. LogProb top-K=16 (student top-K OPD) is their default (Table 2); grpo_fast only
#      implements sampled-token reverse-KL (--opd_pure). For a literal repro, run their
#      verl job instead (LOG_PROB_TOP_K=16, TOP_K_STRATEGY=only_stu).
#   2. Teacher: verl serves teacher on separate vLLM workers; we load teacher learner-side
#      (DeepSpeed ZeRO-3), same as other open-instruct OPD runs.
#   3. Data: paper uses BytedTsinghua-SIA/DAPO-Math-17k parquet; below defaults to
#      hamishivi/DAPO-Math-17k-Processed_filtered (~12.6k prompts). Override DATASET if
#      you pin their exact parquet on WEKA/HF.
#   4. Prompt template may differ from their DAPO "Answer:" suffix; match their parquet
#      fields or preprocess before claiming number parity.
#
# Usage:
#   ./scripts/train/build_image_and_launch.sh scripts/train/qwen/qwen3_17b_thunlp_opd_dapo_math_8gpu.sh
#
# Optional env overrides:
#   TEACHER_MODEL, MODEL, DATASET, NUM_TRAIN_PROMPTS, NUM_NODES, NUM_GPUS_PER_NODE
set -euo pipefail

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"
shift

MODEL="${MODEL:-Qwen/Qwen3-1.7B-Base}"
TOKENIZER="${TOKENIZER:-$MODEL}"
TEACHER_MODEL="${TEACHER_MODEL:-lllyx/Qwen3-4B-Base-GRPO}"
DATASET="${DATASET:-hamishivi/DAPO-Math-17k-Processed_filtered}"
# Paper: 1 epoch over full train split, 4 completions per prompt (Table 2 rollout n=4).
NUM_TRAIN_PROMPTS="${NUM_TRAIN_PROMPTS:-12643}"
ROLLOUTS_PER_PROMPT="${ROLLOUTS_PER_PROMPT:-4}"
TOTAL_EPISODES="${TOTAL_EPISODES:-$((NUM_TRAIN_PROMPTS * ROLLOUTS_PER_PROMPT))}"
GLOBAL_PROMPT_BATCH="${GLOBAL_PROMPT_BATCH:-64}"

EXP_NAME="${EXP_NAME:-qwen3_17b_thunlp_opd_dapo_math}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
PRIORITY="${PRIORITY:-urgent}"
NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
# 1 node: half learners / half vLLM (THUNLP opd.sh default is 4 GPUs; scale batch via GLOBAL_PROMPT_BATCH).
NUM_LEARNERS_PER_NODE="${NUM_LEARNERS_PER_NODE:-4}"
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-4}"

uv run python mason.py \
    --task_name "$EXP_NAME" \
    --description "$RUN_NAME: THUNLP-style OPD Qwen3-1.7B-Base <- Qwen3-4B-Base-GRPO on DAPO math" \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority "$PRIORITY" \
    --pure_docker_mode \
    --image "$BEAKER_IMAGE" \
    --min_runtime 4h \
    --auto_resume \
    --num_nodes "$NUM_NODES" \
    --max_retries 0 \
    --timeout 12h \
    --gpus "$NUM_GPUS_PER_NODE" \
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
    --dataset_mixer_list "$DATASET" 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 1024 \
    --response_length 7168 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout "$GLOBAL_PROMPT_BATCH" \
    --num_samples_per_prompt_rollout "$ROLLOUTS_PER_PROMPT" \
    --async_steps 1 \
    --inflight_updates false \
    --opd_teacher_model_name_or_path "$TEACHER_MODEL" \
    --opd_kl_coef 1.0 \
    --opd_pure \
    --filter_zero_std_samples false \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --temperature 1.0 \
    --vllm_top_p 1.0 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --total_episodes "$TOTAL_EPISODES" \
    --num_epochs 1 \
    --deepspeed_stage 3 \
    --num_learners_per_node "$NUM_LEARNERS_PER_NODE" \
    --vllm_num_engines "$VLLM_NUM_ENGINES" \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.7 \
    --load_ref_policy false \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --loss_fn dapo \
    --advantage_normalization_type centered \
    --chat_template qwen_instruct_user_boxed_math \
    --mask_truncated_completions false \
    --gradient_checkpointing \
    --local_eval_every -1 \
    --eval_on_step_0 false \
    --save_freq 20 \
    --checkpoint_state_freq 20 \
    --keep_last_n_checkpoints 3 \
    --output_dir /output \
    --with_tracking \
    --wandb_entity allenai-team1 \
    --wandb_project opd \
    --seed 42 \
    --vllm_sync_backend gloo \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs_on_weka false "$@"
