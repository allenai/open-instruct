#!/bin/bash
# Eval script for SFT tokenization test models

set -e

# Update oe-eval-internal to the required branch
echo "Updating oe-eval-internal to yanhongl/hybrid-latest..."
cd oe-eval-internal
git fetch origin
git checkout yanhongl/hybrid-latest
git pull origin yanhongl/hybrid-latest
cd ..

BASE_PATH="/weka/oe-adapt-default/nathanl/checkpoints"

MODELS=(
    # "TEST_HYBRIC_SFT_LARGER_LR1e-4"
    # "TEST_HYBRIC_SFT_LARGER_LR5e-5"
    # "TEST_HYBRIC_SFT_LARGER_LR2.5e-5"
    # "TEST_HYBRIC_SFT_LARGER_LR4.5e-5_seed42"
    "TEST_HYBRIC_SFT_LARGER_LR1e-5"
)

for MODEL in "${MODELS[@]}"; do
    GCS_PATH="${BASE_PATH}/${MODEL}/step46412-hf"
    MODEL_NAME="sft-tokenization-test-${MODEL}"

    echo "====================================="
    echo "Running evals for: ${MODEL}"
    echo "Path: ${GCS_PATH}"
    echo "====================================="

    # Batch 1: gpu_multiplier 2
    uv run scripts/submit_eval_jobs.py \
        --model_name "${MODEL_NAME}" \
        --location "${GCS_PATH}" \
        --cluster ai2/jupiter ai2/ceres \
        --is_tuned \
        --workspace ai2/olmo-instruct \
        --priority urgent \
        --preemptible \
        --use_hf_tokenizer_template \
        --beaker_image yanhongl/oe_eval_olmo3_devel_v5 \
        --oe_eval_tasks "gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,omega_500:0-shot-chat_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,ifeval_ood::tulu-thinker-deepseek" \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 32768 \
        --process_output r1_style \
        --skip_oi_evals

    # Batch 2: gpu_multiplier 2
    uv run scripts/submit_eval_jobs.py \
        --model_name "${MODEL_NAME}" \
        --location "${GCS_PATH}" \
        --cluster ai2/jupiter ai2/ceres \
        --is_tuned \
        --workspace ai2/olmo-instruct \
        --priority urgent \
        --gpu_multiplier 2 \
        --preemptible \
        --use_hf_tokenizer_template \
        --beaker_image yanhongl/oe_eval_olmo3_devel_v5 \
        --oe_eval_tasks "bbh:cot::hamish_zs_reasoning_deepseek_v2,mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek" \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 32768 \
        --process_output r1_style \
        --skip_oi_evals

    echo "Completed evals for: ${MODEL}"
    echo ""
done

echo "All eval jobs submitted!"
