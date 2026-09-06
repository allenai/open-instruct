#!/bin/bash

# Evaluate one Qwen3.5 model with the same local math verifier and fixed datasets
# used by the paired 2B OPD/verifier runs. Example:
#   EVAL_MODE=sampled MODEL_LABEL=verifier_final MODEL=/weka/path/to/model \
#     ./scripts/general_agent/terminal/rl/qwen35_math_posthoc_eval.sh IMAGE DAPO_DATASET PATCH_DATASET
set -euo pipefail

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image> <dapo-split-dataset> <code-patch-dataset>}"
DAPO_SPLIT_DATASET="${2:?Usage: $0 <beaker-image> <dapo-split-dataset> <code-patch-dataset>}"
CODE_PATCH_DATASET="${3:?Usage: $0 <beaker-image> <dapo-split-dataset> <code-patch-dataset>}"

MODEL="${MODEL:?Set MODEL to a Hugging Face model ID or Weka checkpoint path}"
MODEL_LABEL="${MODEL_LABEL:?Set MODEL_LABEL to a short experiment-safe name}"
TOKENIZER="${TOKENIZER:-$MODEL}"
VLLM_MODEL="${VLLM_MODEL:-$MODEL}"
EVAL_MODE="${EVAL_MODE:-sampled}"
WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-urgent}"

case "$EVAL_MODE" in
    greedy)
        TEMPERATURE=0.0
        EVAL_PASS_AT_K=1
        ;;
    sampled)
        TEMPERATURE=1.0
        EVAL_PASS_AT_K=8
        ;;
    *)
        echo "EVAL_MODE must be 'greedy' or 'sampled', got '$EVAL_MODE'" >&2
        exit 2
        ;;
esac

EXP_NAME="qwen35_math_posthoc_${MODEL_LABEL}_${EVAL_MODE}"

uv run python mason.py \
    --task_name "$EXP_NAME" \
    --description "$EXP_NAME: matched DAPO/AIME/BRUMO evaluation only" \
    --cluster "$CLUSTER" \
    --workspace "$WORKSPACE" \
    --priority "$PRIORITY" \
    --pure_docker_mode \
    --image "$BEAKER_IMAGE" \
    --beaker_datasets "/patch:$CODE_PATCH_DATASET" "/dapo:$DAPO_SPLIT_DATASET" \
    --min_runtime 2h \
    --no_auto_resume \
    --num_nodes 1 \
    --max_retries 0 \
    --timeout 6h \
    --gpus 8 \
    --auto_checkpoint_state_dir "" \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    --env VLLM_DISABLE_COMPILE_CACHE=1 \
    --env VLLM_USE_V1=1 \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --no_auto_dataset_cache \
    -- \
cp /patch/open_instruct/data_loader.py \
    /patch/open_instruct/ground_truth_utils.py \
    /patch/open_instruct/grpo_fast.py \
    /patch/open_instruct/grpo_utils.py \
    /stage/open_instruct/ \
\&\& source configs/beaker_configs/ray_node_setup.sh \
\&\& uv run open_instruct/grpo_fast.py \
    --exp_name "$EXP_NAME" \
    --model_name_or_path "$MODEL" \
    --tokenizer_name_or_path "$TOKENIZER" \
    --vllm_model_name_or_path "$VLLM_MODEL" \
    --dataset_mixer_list /dapo/train.jsonl 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list \
        /dapo/eval.jsonl 1.0 \
        mnoukhov/aime_2025_openinstruct 1.0 \
        mnoukhov/brumo_2025_openinstruct 1.0 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 18432 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 1 \
    --total_episodes 4 \
    --async_steps 1 \
    --filter_zero_std_samples false \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --remap_verifier dapo_math_holdout=math,math_aime_2025=math,math_brumo_2025=math \
    --temperature "$TEMPERATURE" \
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
    --chat_template qwen_instruct_user_boxed_math \
    --eval_pass_at_k "$EVAL_PASS_AT_K" \
    --local_eval_every 1 \
    --synchronous_local_eval true \
    --eval_on_step_0 true \
    --eval_only true \
    --final_eval_timeout 7200 \
    --save_freq -1 \
    --checkpoint_state_freq -1 \
    --output_dir /output \
    --with_tracking \
    --wandb_entity allenai-team1 \
    --wandb_project opd \
    --seed 42 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs_on_weka false \
    --hf_entity allenai
