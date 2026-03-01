#!/bin/bash
# Retry script for hanging/failed evals. Uses _repeat_3 to avoid overwriting.
# Easy to add more tasks to RETRY_TASKS below.

set -e

BEAKER_IMAGE="yanhongl/oe_eval_olmo3_devel_v7"
HF_LOCATION="allenai/OLMo-3.2-Hybrid-7B-DPO"

# ---- DPO original (repeat_3) — AIME 2024 + livecodebench ----
uv run scripts/submit_eval_jobs.py \
    --model_name "hf-OLMo-3.2-Hybrid-7B-DPO_repeat_3" \
    --location "${HF_LOCATION}" \
    --cluster ai2/jupiter \
    --is_tuned \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --use_hf_tokenizer_template \
    --beaker_image "${BEAKER_IMAGE}" \
    --oe_eval_tasks "aime:zs_cot_r1::pass_at_32_2024_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags" \
    --run_oe_eval_experiments \
    --evaluate_on_weka \
    --run_id placeholder \
    --oe_eval_max_length 32768 \
    --process_output r1_style \
    --skip_oi_evals

# ---- DPO repeat_1 (repeat_3) — AIME 2024 ----
uv run scripts/submit_eval_jobs.py \
    --model_name "hf-OLMo-3.2-Hybrid-7B-DPO_repeat_1_repeat_3" \
    --location "${HF_LOCATION}" \
    --cluster ai2/jupiter \
    --is_tuned \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --use_hf_tokenizer_template \
    --beaker_image "${BEAKER_IMAGE}" \
    --oe_eval_tasks "aime:zs_cot_r1::pass_at_32_2024_deepseek" \
    --run_oe_eval_experiments \
    --evaluate_on_weka \
    --run_id placeholder \
    --oe_eval_max_length 32768 \
    --process_output r1_style \
    --skip_oi_evals

# ---- Add more retries below (e.g. omega) ----

echo "Retry jobs submitted!"
