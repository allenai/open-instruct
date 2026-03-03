#!/bin/bash
set -euo pipefail

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

MODEL_NAME=Qwen/Qwen3-0.6B
TIMESTAMP=$(date +%s)
OUTPUT_DIR=/weka/oe-adapt-default/allennlp/deletable_checkpoint_integration_test/dpo_cache_ckpt_test_${TIMESTAMP}
EXP_NAME_RUN1="dpo-cache-ckpt-integ-run1-${TIMESTAMP}"
EXP_NAME_RUN2="dpo-cache-ckpt-integ-run2-${TIMESTAMP}"

COMMON_ARGS=(
    --model_name_or_path "$MODEL_NAME"
    --tokenizer_name "$MODEL_NAME"
    --use_flash_attn false
    --max_seq_length 1024
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 4
    --learning_rate 5e-07
    --lr_scheduler_type linear
    --warmup_ratio 0.1
    --weight_decay 0.0
    --output_dir "$OUTPUT_DIR"
    --do_not_randomize_output_dir
    --logging_steps 1
    --mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 100
    --add_bos
    --chat_template_name olmo
    --seed 123
    --activation_memory_budget 0.5
    --checkpointing_steps 1
    --keep_last_n_checkpoints 2
    --with_tracking
    --push_to_hub false
    --try_launch_beaker_eval_jobs false
)

wait_for_experiment() {
    local exp_id="$1"
    local run_name="$2"
    echo "Waiting for ${run_name} (experiment ${exp_id}) to finish..."
    beaker experiment await "$exp_id" 0 --index exited --timeout 60m
    beaker experiment await "$exp_id" 1 --index exited --timeout 60m
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

echo "=== Run 1: Train 1 epoch and save checkpoints ==="
run1_output=$(uv run python mason.py \
    --cluster ai2/jupiter \
    --description "Checkpoint integration test run 1: train and save checkpoints" \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 2 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name "$EXP_NAME_RUN1" \
    --num_epochs 1 \
    "${COMMON_ARGS[@]}" 2>&1)

echo "$run1_output"
run1_id=$(extract_experiment_id "$run1_output")
if [[ -z "$run1_id" ]]; then
    echo "ERROR: Could not extract experiment ID from run 1 output."
    exit 1
fi
echo "Run 1 experiment ID: $run1_id"

wait_for_experiment "$run1_id" "Run 1"

echo ""
echo "=== Run 2: Resume from checkpoint and train 2 epochs ==="
run2_output=$(uv run python mason.py \
    --cluster ai2/jupiter \
    --description "Checkpoint integration test run 2: resume from checkpoint" \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 2 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name "$EXP_NAME_RUN2" \
    --num_epochs 2 \
    "${COMMON_ARGS[@]}" 2>&1)

echo "$run2_output"
run2_id=$(extract_experiment_id "$run2_output")
if [[ -z "$run2_id" ]]; then
    echo "ERROR: Could not extract experiment ID from run 2 output."
    exit 1
fi
echo "Run 2 experiment ID: $run2_id"

wait_for_experiment "$run2_id" "Run 2"

echo ""
echo "=== Checkpoint integration test completed ==="
echo "Run 1: https://beaker.org/ex/$run1_id"
echo "Run 2: https://beaker.org/ex/$run2_id"
echo "Check run 2 logs for 'Resumed from checkpoint' message to verify checkpoint resume worked."
