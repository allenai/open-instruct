#!/bin/bash

# Paired math-objective experiment for a Qwen3.5-2B student. Both arms use
# the same deterministic DAPO train split, rollout budget, optimizer, DPPO
# policy loss, and fixed DAPO/AIME/BRUMO evaluations. Set OBJECTIVE=opd for
# teacher-logprob advantages or OBJECTIVE=verifier for grouped math-reward
# advantages. This makes the source of the advantage the controlled variable.
set -euo pipefail

BEAKER_IMAGE="${1:?Usage: OBJECTIVE=opd|verifier $0 <beaker-image> <dapo-split-beaker-dataset>}"
DAPO_SPLIT_DATASET="${2:?Usage: OBJECTIVE=opd|verifier $0 <beaker-image> <dapo-split-beaker-dataset>}"
shift 2

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
TOKENIZER="${TOKENIZER:-$MODEL}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-9B}"
OBJECTIVE="${OBJECTIVE:-verifier}"
RUN_MODE="${RUN_MODE:-full}"
PRIORITY="${PRIORITY:-urgent}"
WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
PATCH_DATASET="${PATCH_DATASET:-}"

BEAKER_DATASETS=(--beaker_datasets "/dapo:$DAPO_SPLIT_DATASET")
PATCH_SETUP=()
if [[ -n "$PATCH_DATASET" ]]; then
    BEAKER_DATASETS+=("/patch:$PATCH_DATASET")
    PATCH_SETUP=(
        cp
        /patch/open_instruct/data_loader.py
        /patch/open_instruct/ground_truth_utils.py
        /patch/open_instruct/grpo_fast.py
        /patch/open_instruct/grpo_utils.py
        /stage/open_instruct/
        '&&'
    )
fi

case "$OBJECTIVE" in
    opd)
        EXP_NAME="${EXP_NAME:-qwen35_2b_opd_math_fixed_dapo_eval_4node}"
        DESCRIPTION="pure OPD from $TEACHER_MODEL"
        OBJECTIVE_ARGS=(
            --opd_teacher_model_name_or_path "$TEACHER_MODEL"
            --opd_kl_coef 1.0
            --opd_pure
        )
        ;;
    verifier)
        EXP_NAME="${EXP_NAME:-qwen35_2b_verifier_dppo_math_fixed_dapo_eval_4node}"
        DESCRIPTION="grouped verifier advantages with DPPO loss"
        OBJECTIVE_ARGS=()
        ;;
    *)
        echo "OBJECTIVE must be 'opd' or 'verifier', got '$OBJECTIVE'" >&2
        exit 2
        ;;
esac

case "$RUN_MODE" in
    smoke)
        NUM_NODES=1
        NUM_LEARNERS_PER_NODE=(4)
        VLLM_NUM_ENGINES=4
        NUM_UNIQUE_PROMPTS=32
        NUM_SAMPLES_PER_PROMPT=2
        ASYNC_STEPS=1
        RESPONSE_LENGTH=8192
        EVAL_RESPONSE_LENGTH=512
        PACK_LENGTH=10240
        TOTAL_EPISODES=64
        LOCAL_EVAL_EVERY=1
        SAVE_FREQ=-1
        CHECKPOINT_STATE_FREQ=-1
        MIN_RUNTIME=1h
        TIMEOUT=2h
        ;;
    full)
        NUM_NODES=4
        NUM_LEARNERS_PER_NODE=(8 8)
        VLLM_NUM_ENGINES=16
        NUM_UNIQUE_PROMPTS=128
        NUM_SAMPLES_PER_PROMPT=2
        ASYNC_STEPS=4
        RESPONSE_LENGTH=16384
        EVAL_RESPONSE_LENGTH=16384
        PACK_LENGTH=18432
        TOTAL_EPISODES=25600
        LOCAL_EVAL_EVERY=20
        SAVE_FREQ=20
        CHECKPOINT_STATE_FREQ=10
        MIN_RUNTIME=4h
        TIMEOUT=12h
        ;;
    *)
        echo "RUN_MODE must be 'smoke' or 'full', got '$RUN_MODE'" >&2
        exit 2
        ;;
esac

RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

uv run python mason.py \
    --task_name "$EXP_NAME" \
    --description "$RUN_NAME: Qwen3.5-2B on fixed DAPO split; $DESCRIPTION" \
    --cluster "$CLUSTER" \
    --workspace "$WORKSPACE" \
    --priority "$PRIORITY" \
    --pure_docker_mode \
    --image "$BEAKER_IMAGE" \
    "${BEAKER_DATASETS[@]}" \
    --min_runtime "$MIN_RUNTIME" \
    --no_auto_resume \
    --num_nodes "$NUM_NODES" \
    --max_retries 0 \
    --timeout "$TIMEOUT" \
    --gpus 8 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    --env VLLM_DISABLE_COMPILE_CACHE=1 \
    --env VLLM_USE_V1=1 \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --no_auto_dataset_cache \
    -- \
"${PATCH_SETUP[@]}" \
source configs/beaker_configs/ray_node_setup.sh \
\&\& uv run open_instruct/grpo_fast.py \
    --run_name "$RUN_NAME" \
    --exp_name "$EXP_NAME" \
    --model_name_or_path "$MODEL" \
    --tokenizer_name_or_path "$TOKENIZER" \
    --dataset_mixer_list /dapo/train.jsonl 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list \
        /dapo/eval.jsonl 1.0 \
        mnoukhov/aime_2025_openinstruct 1.0 \
        mnoukhov/brumo_2025_openinstruct 1.0 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length "$RESPONSE_LENGTH" \
    --eval_response_length "$EVAL_RESPONSE_LENGTH" \
    --pack_length "$PACK_LENGTH" \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout "$NUM_UNIQUE_PROMPTS" \
    --num_samples_per_prompt_rollout "$NUM_SAMPLES_PER_PROMPT" \
    --async_steps "$ASYNC_STEPS" \
    --inflight_updates true \
    "${OBJECTIVE_ARGS[@]}" \
    --filter_zero_std_samples false \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --remap_verifier dapo_math_holdout=math,math_aime_2025=math,math_brumo_2025=math \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --total_episodes "$TOTAL_EPISODES" \
    --num_epochs 1 \
    --deepspeed_stage 3 \
    --num_learners_per_node "${NUM_LEARNERS_PER_NODE[@]}" \
    --vllm_num_engines "$VLLM_NUM_ENGINES" \
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
    --eval_pass_at_k 1 \
    --local_eval_every "$LOCAL_EVAL_EVERY" \
    --synchronous_local_eval true \
    --final_eval_timeout 1800 \
    --eval_on_step_0 true \
    --save_freq "$SAVE_FREQ" \
    --checkpoint_state_freq "$CHECKPOINT_STATE_FREQ" \
    --keep_last_n_checkpoints 2 \
    --save_traces \
    --save_trainer_logprobs false \
    --rollouts_save_path /weka/oe-adapt-default/allennlp/deletable_rollouts/ \
    --output_dir /output \
    --with_tracking \
    --wandb_entity allenai-team1 \
    --wandb_project opd \
    --seed 42 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs_on_weka false "$@"
