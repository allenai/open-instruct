#!/bin/bash
# Runs inside the Beaker container (invoked by deepcoder_1_5b_eval_checkpoints.sh via mason.py).
# Loops over CHECKPOINTS (space-separated "path:step" pairs, set as an env var by the launcher)
# and runs one --eval_only pass per checkpoint, all within a single job so they share one vLLM
# spin-up instead of paying Beaker's queue/scheduling cost once per checkpoint.
set -e

: "${CHECKPOINTS:?CHECKPOINTS env var (space-separated path:step pairs) must be set}"
: "${WANDB_GROUP_NAME:?WANDB_GROUP_NAME env var must be set}"
: "${EXP_NAME:?EXP_NAME env var must be set}"

BACKEND_ARGS="--fsdp_shard_degree 4 --fsdp_num_replicas 1 --activation_memory_budget 0.5"
# Defaults mirror scripts/train/qwen/deepcoder_1_5b.sh so checkpoint evals are comparable to the
# in-training eval curve: the full 279-problem lcbv5 test set at pass@4.
EVAL_DATASETS="${EVAL_DATASETS:-mnoukhov/deepcoder_lcbv5_test_full 1.0}"
EVAL_PASS_AT_K="${EVAL_PASS_AT_K:-4}"

for entry in $CHECKPOINTS; do
    path="${entry%%:*}"
    step="${entry##*:}"
    run_name="${EXP_NAME}_step${step}_$(date +%Y%m%d_%H%M%S)"
    echo "=== Evaluating checkpoint step ${step}: ${path} (run_name=${run_name}) ==="

    uv run open_instruct/grpo.py \
        --run_name "${run_name}" \
        --exp_name "${EXP_NAME}" \
        --wandb_group_name "${WANDB_GROUP_NAME}" \
        --vllm_top_p 1.0 \
        --eval_temperature 0.6 \
        --eval_pass_at_k ${EVAL_PASS_AT_K} \
        --beta 0.001 \
        --async_steps 2 \
        --active_sampling \
        --inflight_updates \
        --advantage_normalization_type centered \
        --num_samples_per_prompt_rollout 16 \
        --num_unique_prompts_rollout 8 \
        --num_mini_batches 1 \
        --learning_rate 5e-7 \
        --per_device_train_batch_size 1 \
        --temperature 1.0 \
        --dataset_mixer_list mnoukhov/deepcoder_lcbv5_full 1.0 mnoukhov/deepcoder_primeintellect_full 1.0 mnoukhov/deepcoder_taco_full 1.0 \
        --dataset_mixer_list_splits "train" \
        --dataset_mixer_eval_list $EVAL_DATASETS \
        --dataset_mixer_eval_list_splits "train" \
        --max_prompt_token_length 2048 \
        --response_length 32768 \
        --pack_length 34816 \
        --model_name_or_path "${path}" \
        --non_stop_penalty False \
        --total_episodes 64000 \
        ${BACKEND_ARGS} \
        --num_learners_per_node 4 \
        --vllm_num_engines 4 \
        --vllm_tensor_parallel_size 1 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --local_eval_every 100 \
        --save_freq 100 \
        --checkpoint_state_freq 100 \
        --gradient_checkpointing \
        --with_tracking \
        --send_slack_alerts False \
        --vllm_enable_prefix_caching \
        --clip_higher 0.272 \
        --max_grad_norm 1.0 \
        --mask_truncated_completions True \
        --code_api_url "$CODE_API_URL/test_program" \
        --code_pass_rate_reward_threshold 0.0 \
        --load_ref_policy True \
        --keep_last_n_checkpoints 3 \
        --push_to_hub False \
        --eval_only \
        --eval_only_set_checkpoint "${step}"

    echo "=== Done step ${step} ==="
done
