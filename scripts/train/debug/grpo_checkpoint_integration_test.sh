#!/bin/bash
set -euo pipefail

# Integration test for grpo.py (OLMo-core path) checkpoint resume.
# Run 1 trains a few steps and writes a checkpoint; run 2 reuses the same
# --checkpoint_state_dir and must pick up where run 1 stopped.

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

MODEL_NAME=Qwen/Qwen3-1.7B
TIMESTAMP=$(date +%s)
CHECKPOINT_STATE_DIR=/weka/oe-adapt-default/${BEAKER_USER}/deletable_checkpoint_states/grpo_resume_test_${TIMESTAMP}/tmp-1d
EXP_NAME_RUN1="grpo-ckpt-integ-run1-${TIMESTAMP}"
EXP_NAME_RUN2="grpo-ckpt-integ-run2-${TIMESTAMP}"

# 8 prompts * 4 samples = 32 episodes per training step.
# Run 1: total_episodes=192 -> 6 steps; checkpoint_state_freq=2 -> ckpts at 2,4,6.
# Run 2: total_episodes=384 -> 12 steps; resume should start above step 0.
COMMON_ARGS=(
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64
    --dataset_mixer_list_splits train
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16
    --dataset_mixer_eval_list_splits train
    --max_prompt_token_length 512
    --response_length 512
    --pack_length 1024
    --per_device_train_batch_size 1
    --num_unique_prompts_rollout 8
    --num_samples_per_prompt_rollout 4
    --model_name_or_path "$MODEL_NAME"
    --stop_strings "</answer>"
    --apply_r1_style_format_reward
    --apply_verifiable_reward true
    --temperature 0.7
    --inflight_updates True
    --ground_truths_key ground_truth
    --chat_template_name r1_simple_chat_postpend_think
    --learning_rate 3e-7
    --deepspeed_stage 2
    --num_epochs 1
    --num_learners_per_node 1
    --vllm_tensor_parallel_size 1
    --beta 0.0
    --load_ref_policy true
    --seed 3
    --local_eval_every 1000
    --vllm_sync_backend gloo
    --vllm_gpu_memory_utilization 0.3
    --save_traces
    --vllm_enforce_eager
    --gradient_checkpointing
    --push_to_hub false
    --async_steps 1
    --single_gpu_mode
    --checkpoint_state_dir "$CHECKPOINT_STATE_DIR"
    --checkpoint_state_freq 2
    --output_dir "$CHECKPOINT_STATE_DIR/output"
)

wait_for_experiment() {
    local exp_id="$1"
    local run_name="$2"
    echo "Waiting for ${run_name} (experiment ${exp_id}) to finish..."
    beaker experiment await "$exp_id" 0 --index exited --timeout 60m
    exit_code=$(beaker experiment get "$exp_id" --format json | jq -r '.[0].jobs[0].status.exitCode')
    if [ "$exit_code" != "0" ]; then
        echo "ERROR: ${run_name} failed with exit code ${exit_code}"
        return 1
    fi
    echo "${run_name} completed successfully."
}

extract_experiment_id() {
    local output="$1"
    echo "$output" | grep -o 'https://beaker.org/ex/[A-Za-z0-9_-]*' | tail -1 | sed 's|https://beaker.org/ex/||'
}

launch_run() {
    local description="$1"
    local exp_name="$2"
    local total_episodes="$3"
    uv run python mason.py \
        --cluster ai2/jupiter \
        --cluster ai2/saturn \
        --image "$BEAKER_IMAGE" \
        --description "$description" \
        --pure_docker_mode \
        --workspace ai2/open-instruct-dev \
        --priority urgent \
        --num_nodes 1 \
        --max_retries 0 \
        --timeout 30m \
        --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        --budget ai2/oe-omai \
        --gpus 1 \
        --no_auto_dataset_cache \
        --artifact_ttl 1d \
        -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo.py \
        --exp_name "$exp_name" \
        --total_episodes "$total_episodes" \
        "${COMMON_ARGS[@]}" 2>&1
}

echo "=== Run 1: train 6 steps and save checkpoints to $CHECKPOINT_STATE_DIR ==="
run1_output=$(launch_run "GRPO checkpoint integration test run 1" "$EXP_NAME_RUN1" 192)
echo "$run1_output"
run1_id=$(extract_experiment_id "$run1_output")
[[ -n "$run1_id" ]] || { echo "ERROR: Could not extract run 1 experiment ID."; exit 1; }
echo "Run 1 experiment ID: $run1_id"
wait_for_experiment "$run1_id" "Run 1"

echo ""
echo "=== Run 2: resume from $CHECKPOINT_STATE_DIR and train to 12 steps ==="
run2_output=$(launch_run "GRPO checkpoint integration test run 2 (resume)" "$EXP_NAME_RUN2" 384)
echo "$run2_output"
run2_id=$(extract_experiment_id "$run2_output")
[[ -n "$run2_id" ]] || { echo "ERROR: Could not extract run 2 experiment ID."; exit 1; }
echo "Run 2 experiment ID: $run2_id"
wait_for_experiment "$run2_id" "Run 2"

echo ""
echo "=== GRPO checkpoint integration test completed ==="
echo "Run 1: https://beaker.org/ex/$run1_id"
echo "Run 2: https://beaker.org/ex/$run2_id"
echo "Verify run 2 logs contain a 'Loading checkpoint from' / 'Loaded checkpoint' message and that training resumes above step 0."
