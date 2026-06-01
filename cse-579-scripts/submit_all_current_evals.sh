#!/bin/bash
# Submit oe-eval jobs for all currently-available checkpoints of the four active
# CSE-579 experiments (the two warmup runs + the two GFPO runs), as of 2026-06-01.
#
# The training runs' auto-launched evals are not appearing in ai2/olmo-instruct,
# so we submit manually with the known-good config (Jacob's tokenizer_path that
# sidesteps the transformers-version tokenizer bug). Same submit_eval_jobs.py
# invocation as cse-579-scripts/submit_lenshape_qwen_eval_jobs.sh.
#
# Checkpoint dirs/steps were discovered via `aws s3 ls ... --profile WEKA`.
# Re-run safely as runs progress; it just resubmits (oe-eval dedups by name on
# the results side, but avoid spamming — comment out steps already evaluated).
#
# Priority defaults to normal (29 checkpoints x 5 tasks is a lot of jobs; don't
# hog urgent). Override with PRIORITY=urgent.

set -euo pipefail

PRIORITY=${PRIORITY:-normal}
CKPT_ROOT=/weka/oe-adapt-default/allennlp/deletable_checkpoint/ianm
current_evals="alpaca_eval_v3::hamish_zs_reasoning_deepseek,minerva_math_500::hamish_zs_reasoning,ifbench::tulu,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2025_deepseek"

# Each entry: "<checkpoint_dir_basename> <space-separated step numbers>"
RUNS=(
  "lenshape_qwen_4b_base_mixed_linear_p1.0_wlinear__1__1780178009_checkpoints 100 200 300 400 500 600 700 800 900 1000"
  "lenshape_qwen_4b_base_mixed_linear_p1.0_wsolve_rate__1__1780178056_checkpoints 100 200 300 400 500 600 700 800 900 1000"
  "gfpo_qwen_4b_base_mixed_shortest_g16k8__1__1780178181_checkpoints 100 200 300 400 500 600"
  "gfpo_qwen_4b_base_mixed_token_efficiency_g16k8__1__1780274849_checkpoints 100 200 300"
)

MODEL_PATHS=()
for entry in "${RUNS[@]}"; do
  read -r dir steps <<< "$entry"
  for step in $steps; do
    MODEL_PATHS+=("$CKPT_ROOT/$dir/step_$step")
  done
done

echo "Submitting evals for ${#MODEL_PATHS[@]} checkpoints (priority=$PRIORITY)..."

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  BASENAME=$(basename "$MODEL_PATH")          # step_N
  EXPERIMENT_NAME=$(basename "$(dirname "$MODEL_PATH")" _checkpoints)
  MODEL_NAME="${EXPERIMENT_NAME}_${BASENAME}"

  echo "Submitting eval for: $MODEL_NAME"
  uv run python scripts/submit_eval_jobs.py \
      --model_name "${MODEL_NAME}" \
      --location "$MODEL_PATH" \
      --cluster ai2/saturn ai2/ceres \
      --is_tuned \
      --workspace ai2/olmo-instruct \
      --priority "$PRIORITY" \
      --preemptible \
      --use_hf_tokenizer_template \
      --run_oe_eval_experiments \
      --evaluate_on_weka \
      --run_id placeholder \
      --oe_eval_max_length 32768 \
      --process_output r1_style \
      --skip_oi_evals \
      --tokenizer_path /weka/oe-adapt-default/jacobm/repos/cse-579/tokenizers/qwen3-olmo-thinker-eos-old-transformers \
      --oe_eval_tasks $current_evals
done
