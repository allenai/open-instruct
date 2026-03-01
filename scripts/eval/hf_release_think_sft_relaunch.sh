#!/bin/bash
# Relaunch ALL evals for Think SFT after chat template fix.
# Uses "v2" suffix to avoid overwriting old results.

set -e

BEAKER_IMAGE="yanhongl/oe_eval_olmo3_devel_v6"
MODEL_NAME="hf-Olmo-3.2-Hybrid-7B-Think-SFT-backup-3"
HF_LOCATION="allenai/Olmo-3.2-Hybrid-7B-Think-SFT-backup" # note changed bc worry about original checkpoint upload

BATCH1_TASKS="gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,minerva_math::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,ifeval_ood::tulu-thinker-deepseek"
BATCH2_TASKS="bbh:cot::hamish_zs_reasoning_deepseek_v2,mmlu:cot::hamish_zs_reasoning_deepseek,omega:0-shot-chat_deepseek,popqa::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek"
TRIPLE_TASKS="omega:0-shot-chat_deepseek,minerva_math::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags"

echo "====================================="
echo "Relaunching all evals for Think SFT (v2 - chat template fix)"
echo "====================================="

# ---- Run 1 (original) ----
uv run scripts/submit_eval_jobs.py \
    --model_name "${MODEL_NAME}" \
    --location "${HF_LOCATION}" \
    --cluster ai2/jupiter \
    --is_tuned \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --use_hf_tokenizer_template \
    --beaker_image "${BEAKER_IMAGE}" \
    --oe_eval_tasks "${BATCH1_TASKS}" \
    --run_oe_eval_experiments \
    --evaluate_on_weka \
    --run_id placeholder \
    --oe_eval_max_length 32768 \
    --process_output r1_style \
    --skip_oi_evals

uv run scripts/submit_eval_jobs.py \
    --model_name "${MODEL_NAME}" \
    --location "${HF_LOCATION}" \
    --cluster ai2/jupiter \
    --is_tuned \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --gpu_multiplier 2 \
    --preemptible \
    --use_hf_tokenizer_template \
    --beaker_image "${BEAKER_IMAGE}" \
    --oe_eval_tasks "${BATCH2_TASKS}" \
    --run_oe_eval_experiments \
    --evaluate_on_weka \
    --run_id placeholder \
    --oe_eval_max_length 32768 \
    --process_output r1_style \
    --skip_oi_evals

# Commented out below to test -backup model
# ---- Run 2 (repeat_1) ----
# echo "  -> Repeat 1 (all evals): ${MODEL_NAME}_repeat_1"

# uv run scripts/submit_eval_jobs.py \
#     --model_name "${MODEL_NAME}_repeat_1" \
#     --location "${HF_LOCATION}" \
#     --cluster ai2/jupiter \
#     --is_tuned \
#     --workspace ai2/olmo-instruct \
#     --priority urgent \
#     --preemptible \
#     --use_hf_tokenizer_template \
#     --beaker_image "${BEAKER_IMAGE}" \
#     --oe_eval_tasks "${BATCH1_TASKS}" \
#     --run_oe_eval_experiments \
#     --evaluate_on_weka \
#     --run_id placeholder \
#     --oe_eval_max_length 32768 \
#     --process_output r1_style \
#     --skip_oi_evals

# uv run scripts/submit_eval_jobs.py \
#     --model_name "${MODEL_NAME}_repeat_1" \
#     --location "${HF_LOCATION}" \
#     --cluster ai2/jupiter \
#     --is_tuned \
#     --workspace ai2/olmo-instruct \
#     --priority urgent \
#     --gpu_multiplier 2 \
#     --preemptible \
#     --use_hf_tokenizer_template \
#     --beaker_image "${BEAKER_IMAGE}" \
#     --oe_eval_tasks "${BATCH2_TASKS}" \
#     --run_oe_eval_experiments \
#     --evaluate_on_weka \
#     --run_id placeholder \
#     --oe_eval_max_length 32768 \
#     --process_output r1_style \
#     --skip_oi_evals

# # ---- Run 3 (repeat_2) — omega, math, livecodebench only ----
# echo "  -> Repeat 2 (omega/math/livecodebench only): ${MODEL_NAME}_repeat_2"

# uv run scripts/submit_eval_jobs.py \
#     --model_name "${MODEL_NAME}_repeat_2" \
#     --location "${HF_LOCATION}" \
#     --cluster ai2/jupiter \
#     --gpu_multiplier 2 \
#     --is_tuned \
#     --workspace ai2/olmo-instruct \
#     --priority urgent \
#     --preemptible \
#     --use_hf_tokenizer_template \
#     --beaker_image "${BEAKER_IMAGE}" \
#     --oe_eval_tasks "${TRIPLE_TASKS}" \
#     --run_oe_eval_experiments \
#     --evaluate_on_weka \
#     --run_id placeholder \
#     --oe_eval_max_length 32768 \
#     --process_output r1_style \
#     --skip_oi_evals

echo "All Think SFT v2 eval jobs submitted!"
