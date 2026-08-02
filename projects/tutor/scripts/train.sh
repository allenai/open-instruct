#!/bin/bash
# The real run: multi-turn tutoring, a served student, a served judge.
#
# Start ./serve_env.sh first and leave it running. This script trains on the
# remaining GPUs.
#
#   PARTNER_URL=http://localhost:8001/v1 JUDGE_URL=http://localhost:8002/v1 \
#   ./projects/tutor/scripts/train.sh
#
# READ THIS BEFORE PICKING A POLICY SIZE. open-instruct's GRPO is
# FULL-PARAMETER. `use_peft` is declared in model_utils.py and referenced
# nowhere in grpo_fast.py, so there is no LoRA escape hatch: a 3B policy needs
# weights, gradients, optimiser state and activations resident beside vLLM.
# That does not fit on one 80GB card next to an environment and a judge. Either
# run the policy at 0.6-1.5B on one GPU, or give the trainer its own card or
# two and serve the student and judge elsewhere - which is what this script
# assumes. If you need a 3B policy on a single card, that is the one thing the
# TRL path did better and it is documented in projects/tutor/README.md.

set -euo pipefail

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1

MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
DATA=${DATA:-data/tutor_train.jsonl}
PARTNER_MODEL=${PARTNER_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
PARTNER_URL=${PARTNER_URL:-http://localhost:8001/v1}
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
JUDGE_URL=${JUDGE_URL:-http://localhost:8002/v1}
TURNS=${TURNS:-3}
GROUP=${GROUP:-8}
STEPS=${STEPS:-250}
OUT=${OUT:-output/tutor_v5}

# 250 steps is more than enough. Every run in this project's history plateaued
# by roughly step 150, and the last 50 steps of one of them actively degraded.

uv run python open_instruct/grpo_fast.py \
    --dataset_mixer_list "$DATA" 1.0 \
    --dataset_mixer_list_splits train \
    --model_name_or_path "$MODEL" \
    --reward_plugins projects.tutor.plugin \
    --group_scorer "tutor:judge_model=$JUDGE_MODEL,judge_base_url=$JUDGE_URL" \
    --apply_verifiable_reward false \
    --tools tutor_student \
    --tool_configs "{\"tutor_student\": {\"model\": \"$PARTNER_MODEL\", \"base_url\": \"$PARTNER_URL\", \"max_turns\": $TURNS, \"director\": \"projects.tutor.student.StudentDirector\"}}" \
    --max_steps "$TURNS" \
    --mask_tool_use true \
    --max_prompt_token_length 1024 \
    --response_length 1536 \
    --pack_length 2816 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout "$GROUP" \
    --per_device_train_batch_size 1 \
    --temperature 1.0 \
    --learning_rate 5e-7 \
    --beta 0.03 \
    --advantage_normalization_type centered \
    --total_episodes $((STEPS * 8 * GROUP)) \
    --deepspeed_stage 2 \
    --gradient_checkpointing \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_sync_backend gloo \
    --seed 0 \
    --save_freq 25 \
    --save_traces \
    --push_to_hub false \
    --output_dir "$OUT"

# --advantage_normalization_type centered, deliberately. The group scorer
# already z-scores each dimension inside the group, so its output has zero mean
# and the centring step is a no-op - which makes the arithmetic exactly
# MO-GRPO's estimator. "standard" would divide by its own std as well; harmless,
# but no longer literally the published equation.
#
# --apply_verifiable_reward false: there is no per-sample verifier here, the
# group scorer IS the reward. The rows still carry dataset "passthrough" so the
# lookup does not warn.

echo "trained -> $OUT"
echo
echo "now run the anchor, which is the number that decides whether this worked:"
echo "  python -m projects.tutor.run_anchor --items data/state_tests/eval_items.jsonl \\"
echo "      --tutor-model $OUT --tutor-url <serve it> \\"
echo "      --student-model $PARTNER_MODEL --student-url $PARTNER_URL \\"
echo "      --out $OUT/anchor_after.json --compare-to $OUT/anchor_before.json"
