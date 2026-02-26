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
# Use v7 for models with transformers latest naming (DPO, HYBRID_INSTRUCT_SFT_8e-5, etc.)
# BEAKER_IMAGE="yanhongl/oe_eval_olmo3_devel_v7"
BEAKER_IMAGE="yanhongl/oe_eval_olmo3_devel_v7"

# Change this to select which path/name config to use (no commenting needed)
MODEL_TYPE="dpo"  # think | think_yarn | think_ablation | instruct | instruct_0217 | instruct_0218 | dpo

# PATHS
# /weka/oe-adapt-default/nathanl/checkpoints/HYBRID_INSTRUCT_SFT_8e-5/step3256-hf
# /weka/oe-adapt-default/nathanl/checkpoints/HYBRID_INSTRUCT_SFT_5e-5/step3256-hf

# YARN MODELS:
# /weka/oe-adapt-default/nathanl/checkpoints/HYBRID_SFT_YARN_LR5e-5/step46412-hf
# /weka/oe-adapt-default/nathanl/checkpoints/HYBRID_SFT_YARN_LR2.5e-5/step46412-hf

# Finbarr DPO models
# /weka/oe-adapt-default/allennlp/deletable_checkpoint/finbarrt/hybrid-7b-DPO-SFT-8e-5-1e-6/

MODELS=(
    # --- Think SFT Models ---
    # "TEST_HYBRIC_SFT_LARGER_LR1e-4"
    # "TEST_HYBRIC_SFT_LARGER_LR5e-5"
    # "TEST_HYBRIC_SFT_LARGER_LR2.5e-5"
    # "TEST_HYBRIC_SFT_LARGER_LR4.5e-5_seed42"
    # "TEST_HYBRIC_SFT_LARGER_LR1e-5"
    # "HYBRID_SFT_YARN_LR5e-5"
    # "HYBRID_SFT_YARN_LR2.5e-5"
    # "ABLATE_HYBRID_THINK_SFT_0210_LR2.5e-5"
    # "ABLATE_HYBRID_THINK_SFT_0210_5e-5"

    # --- Original Instruct SFT models (need v7 eval image) ---
    # "HYBRID_INSTRUCT_SFT_8e-5" # Needs v7 eval image with transformers latest naming 
    # "HYBRID_INSTRUCT_SFT_5e-5" # Needs v7 eval image with transformers latest naming 

    # --- Finbarr DPO models (need v7 eval image) ---
    # "hybrid-7b-DPO-SFT-8e-5-1e-6"

    # --- 0219 DPO sweep models (SFT-2.5e-5 base, need v7 eval image) ---
    "hybrid-7b-DPO-0219-SFT-2.5e-5-LR-1e-6"
    # "hybrid-7b-DPO-0219-SFT-2.5e-5-LR-2e-6"
    # "hybrid-7b-DPO-0219-SFT-2.5e-5-LR-8.5e-7"
    # "hybrid-7b-DPO-0219-SFT-2.5e-5-LR-7e-7"
    # "hybrid-7b-DPO-0219-SFT-2.5e-5-LR-5e-7"
    # "hybrid-7b-DPO-0219-SFT-2.5e-5-LR-2.5e-7"

    # --- 0217 instruct SFT models (step3256) ---
    # "HYBRID_INSTRUCT_SFT_0217_8e-5"
    # "HYBRID_INSTRUCT_SFT_0217_5e-5"
    # "HYBRID_INSTRUCT_SFT_0217_2.5e-5"
    # "HYBRID_INSTRUCT_SFT_0217_1e-4"
    # "HYBRID_INSTRUCT_SFT_0217_6e-5"
    # "HYBRID_INSTRUCT_SFT_0217_3e-5"
    # "HYBRID_INSTRUCT_SFT_0217_1.5e-5"

    # --- 0218 instruct SFT models (step3256) ---
    # "HYBRID_INSTRUCT_SFT_0218_9e-5"
    # "HYBRID_INSTRUCT_SFT_0218_8e-5"
    # "HYBRID_INSTRUCT_SFT_0218_6e-5"
    # "HYBRID_INSTRUCT_SFT_0218_5e-5"
    # "HYBRID_INSTRUCT_SFT_0218_2.5e-5"
    # "HYBRID_INSTRUCT_SFT_0218_1e-4"

)

for MODEL in "${MODELS[@]}"; do
    case "${MODEL_TYPE}" in
        think)
            GCS_PATH="${BASE_PATH}/${MODEL}/step46412-hf-tokenizer-fix"
            MODEL_NAME="0216-extra-2-${MODEL}"
            ;;
        instruct_0217)
            GCS_PATH="${BASE_PATH}/${MODEL}/step3256-hf"
            # GCS_PATH="${BASE_PATH}/${MODEL}/step3256-fix-hf"
            MODEL_NAME="0217-instruct-${MODEL}"
            ;;
        instruct_0218)
            GCS_PATH="${BASE_PATH}/${MODEL}/step3256-hf"
            # GCS_PATH="${BASE_PATH}/${MODEL}/step3256-fix-hf"
            MODEL_NAME="0218-instruct-v2-${MODEL}"
            ;;
        think_yarn)
            GCS_PATH="${BASE_PATH}/${MODEL}/step43110-hf"
            MODEL_NAME="0210-${MODEL}"
            ;;
        think_ablation)
            GCS_PATH="${BASE_PATH}/${MODEL}/step46412-hf"
            MODEL_NAME="0208-${MODEL}"
            ;;
        dpo)
            GCS_PATH="/weka/oe-adapt-default/allennlp/deletable_checkpoint/nathanl/${MODEL}/"
            # GCS_PATH="/weka/oe-adapt-default/allennlp/deletable_checkpoint/finbarrt/${MODEL}/"
            MODEL_NAME="dpo-0219-${MODEL}"
            ;;
        instruct)
            GCS_PATH="${BASE_PATH}/${MODEL}/step3256-fix-hf"
            # GCS_PATH="${BASE_PATH}/${MODEL}/step3256-hf"
            MODEL_NAME="instruct-sft-hybrid-tok-0218r-${MODEL}"
            # MODEL_NAME="instruct-sft-hybrid-tok-0207-${MODEL}"
            ;;
        *)
            echo "Unknown MODEL_TYPE: ${MODEL_TYPE}"
            exit 1
            ;;
    esac

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
        --beaker_image "${BEAKER_IMAGE}" \
        --oe_eval_tasks "gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,omega_500:0-shot-chat_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,ifeval_ood::tulu-thinker-deepseek" \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 32768 \
        --process_output r1_style \
        --skip_oi_evals

        #  commented out agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek bc broken
        # --oe_eval_tasks "gpqa:0shot_cot::qwen3-instruct,ifeval::hamish_zs_reasoning_deepseek" \

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
        --beaker_image "${BEAKER_IMAGE}" \
        --oe_eval_tasks "bbh:cot::hamish_zs_reasoning_deepseek_v2,mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek" \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 32768 \
        --process_output r1_style \
        --skip_oi_evals

    # ---- BEGIN: Repeated evals for variance estimation (GPQA, LiveCodeBench, IFEval) ----
    # Run these 3 evals 2 more times with different names to measure variance.
    # Comment out this entire block to disable repeated runs.
    for REPEAT in 1 2; do
    # for REPEAT in 1; do
        REPEAT_MODEL_NAME="${MODEL_NAME}_repeat_${REPEAT}"
        echo "  -> Repeat ${REPEAT}: ${REPEAT_MODEL_NAME}"

        uv run scripts/submit_eval_jobs.py \
            --model_name "${REPEAT_MODEL_NAME}" \
            --location "${GCS_PATH}" \
            --cluster ai2/jupiter ai2/ceres \
            --is_tuned \
            --workspace ai2/olmo-instruct \
            --priority urgent \
            --preemptible \
            --use_hf_tokenizer_template \
            --beaker_image "${BEAKER_IMAGE}" \
            --oe_eval_tasks "gpqa:0shot_cot::qwen3-instruct,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,ifeval::hamish_zs_reasoning_deepseek" \
            --run_oe_eval_experiments \
            --evaluate_on_weka \
            --run_id placeholder \
            --oe_eval_max_length 32768 \
            --process_output r1_style \
            --skip_oi_evals
    done
    # ---- END: Repeated evals for variance estimation ----

    echo "Completed evals for: ${MODEL}"
    echo ""
done

echo "All eval jobs submitted!"
