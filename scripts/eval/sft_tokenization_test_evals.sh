#!/bin/bash
# Eval script for SFT tokenization test models

set -e

# NOTE: Skipping git pull to preserve local launch_utils.py changes
# (added runtime transformers fork install for OLMo 3.5 Hybrid support)
# cd oe-eval-internal
# git fetch origin
# git checkout yanhongl/hybrid-latest
# git pull origin yanhongl/hybrid-latest
# cd ..

BASE_PATH="/weka/oe-adapt-default/nathanl/checkpoints"
BASE_PATH_OLMO_CORE="/weka/oe-training-default/ai2-llm/checkpoints/nathanl/olmo-sft"

# /weka/oe-adapt-default/nathanl/checkpoints/HYBRID_INSTRUCT_SFT_8e-5/step3256-hf
# /weka/oe-adapt-default/nathanl/checkpoints/HYBRID_INSTRUCT_SFT_5e-5/step3256-hf

# YARN MODELS:
# /weka/oe-adapt-default/nathanl/checkpoints/HYBRID_SFT_YARN_LR5e-5/step46412-hf
# /weka/oe-adapt-default/nathanl/checkpoints/HYBRID_SFT_YARN_LR2.5e-5/step46412-hf

MODELS=(
    # "TEST_HYBRIC_SFT_LARGER_LR1e-4"
    # "TEST_HYBRIC_SFT_LARGER_LR5e-5"
    # "TEST_HYBRIC_SFT_LARGER_LR2.5e-5"
    # "HYBRID_SFT_YARN_LR5e-5"
    # "HYBRID_SFT_YARN_LR2.5e-5"
    # "ABLATE_HYBRID_THINK_SFT_0210_LR2.5e-5"
    # "ABLATE_HYBRID_THINK_SFT_0210_5e-5"
    # "TEST_HYBRIC_SFT_LARGER_LR4.5e-5_seed42"
    # "TEST_HYBRIC_SFT_LARGER_LR1e-5"
    "HYBRID_INSTRUCT_SFT_8e-5"
    "HYBRID_INSTRUCT_SFT_5e-5"
)

for MODEL in "${MODELS[@]}"; do
    # THINK MODELS
    # GCS_PATH="${BASE_PATH}/${MODEL}/step46412-hf-tokenizer-fix"
    # MODEL_NAME="0208-${MODEL}"

    # Think models -- yarn
    # GCS_PATH="${BASE_PATH}/${MODEL}/step43110-hf"
    # MODEL_NAME="0210-${MODEL}"

    # Think models -- ablation
    # GCS_PATH="${BASE_PATH}/${MODEL}/step46412-hf"
    # MODEL_NAME="0208-${MODEL}"


    # INSTRUCT MODELS
    # GCS_PATH="${BASE_PATH}/${MODEL}/step3256-hf"
    # MODEL_NAME="instruct-sft-hybrid-tok-0207-${MODEL}"
    GCS_PATH="${BASE_PATH_OLMO_CORE}/${MODEL}/step3256"
    MODEL_NAME="instruct-sft-hybrid-tok-0207-${MODEL}-olmocore"

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
        --model_type olmo_core \
        --beaker_image tylerr/oe-eval-olmocore-gdn-v2 \
        --oe_eval_tasks zebralogic::hamish_zs_reasoning_deepseek \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 32768 \
        --process_output r1_style \
        --skip_oi_evals

        # --beaker_image yanhongl/oe_eval_olmo3_devel_v6 \
        # --oe_eval_tasks "gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,omega_500:0-shot-chat_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,ifeval_ood::tulu-thinker-deepseek" \
        # --oe_eval_tasks "gpqa:0shot_cot::qwen3-instruct,ifeval::hamish_zs_reasoning_deepseek" \

    # Batch 2: gpu_multiplier 2
    # uv run scripts/submit_eval_jobs.py \
    #     --model_name "${MODEL_NAME}" \
    #     --location "${GCS_PATH}" \
    #     --cluster ai2/jupiter ai2/ceres \
    #     --is_tuned \
    #     --workspace ai2/olmo-instruct \
    #     --priority urgent \
    #     --gpu_multiplier 2 \
    #     --preemptible \
    #     --use_hf_tokenizer_template \
    #     --beaker_image yanhongl/oe_eval_olmo3_devel_v6 \
    #     --oe_eval_tasks "bbh:cot::hamish_zs_reasoning_deepseek_v2,mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek" \
    #     --run_oe_eval_experiments \
    #     --evaluate_on_weka \
    #     --run_id placeholder \
    #     --oe_eval_max_length 32768 \
    #     --process_output r1_style \
    #     --skip_oi_evals

    echo "Completed evals for: ${MODEL}"
    echo ""
done

echo "All eval jobs submitted!"
