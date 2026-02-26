#!/bin/bash
# Eval script for HuggingFace release models (full eval suites, not 500/lite)
# All evals run 2x, omega/math/livecodebench run 3x for variance estimation.

set -e

BEAKER_IMAGE="yanhongl/oe_eval_olmo3_devel_v7"

# HF model tags and their display names
# Format: "hf-display-name|hf-model-tag"
MODELS=(
    "hf-Olmo-3.2-Hybrid-7B-Think-SFT|allenai/Olmo-3.2-Hybrid-7B-Think-SFT"
    "hf-OLMo-3.2-Hybrid-7B-Instruct-SFT|allenai/OLMo-3.2-Hybrid-7B-Instruct-SFT"
    "hf-OLMo-3.2-Hybrid-7B-DPO|allenai/OLMo-3.2-Hybrid-7B-DPO"
)

# Batch 1 (gpu_multiplier 1): everything except omega, bbh, mmlu, popqa, mbppplus
BATCH1_TASKS="gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,minerva_math::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,ifeval_ood::tulu-thinker-deepseek"

# Batch 2 (gpu_multiplier 2): omega + the larger evals
BATCH2_TASKS="bbh:cot::hamish_zs_reasoning_deepseek_v2,mmlu:cot::hamish_zs_reasoning_deepseek,omega:0-shot-chat_deepseek,popqa::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek"

# 3x tasks (gpu_multiplier 2): omega, math, livecodebench get a third run
TRIPLE_TASKS="omega:0-shot-chat_deepseek,minerva_math::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags"

for ENTRY in "${MODELS[@]}"; do
    MODEL_NAME="${ENTRY%%|*}"
    HF_LOCATION="${ENTRY##*|}"

    echo "====================================="
    echo "Running evals for: ${MODEL_NAME}"
    echo "HF Location: ${HF_LOCATION}"
    echo "====================================="

    # ---- Run 1 of 2 for ALL evals (original name) ----
    # Batch 1: gpu_multiplier 1
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

    # Batch 2: gpu_multiplier 2
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

    # ---- Run 2 of 2 for ALL evals (repeat_1) ----
    echo "  -> Repeat 1 (all evals): ${MODEL_NAME}_repeat_1"

    uv run scripts/submit_eval_jobs.py \
        --model_name "${MODEL_NAME}_repeat_1" \
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
        --model_name "${MODEL_NAME}_repeat_1" \
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

    # ---- Run 3 of 3 for omega, math, livecodebench ONLY (repeat_2) ----
    echo "  -> Repeat 2 (omega/math/livecodebench only): ${MODEL_NAME}_repeat_2"

    uv run scripts/submit_eval_jobs.py \
        --model_name "${MODEL_NAME}_repeat_2" \
        --location "${HF_LOCATION}" \
        --cluster ai2/jupiter \
        --gpu_multiplier 2 \
        --is_tuned \
        --workspace ai2/olmo-instruct \
        --priority urgent \
        --preemptible \
        --use_hf_tokenizer_template \
        --beaker_image "${BEAKER_IMAGE}" \
        --oe_eval_tasks "${TRIPLE_TASKS}" \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 32768 \
        --process_output r1_style \
        --skip_oi_evals

    echo "Completed evals for: ${MODEL_NAME}"
    echo ""
done

echo "All eval jobs submitted!"
