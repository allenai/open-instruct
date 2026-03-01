#!/bin/bash
# Quick single-model, single-eval test script

set -e

BEAKER_IMAGE="yanhongl/oe_eval_olmo3_devel_v6"

uv run scripts/submit_eval_jobs.py \
    --model_name "think-sft-2.5e-5-tokenizer-actually-fixed" \
    --location "/weka/oe-adapt-default/saumyam/checkpoints/TEST_HYBRIC_SFT_LARGER_LR2.5e-5/step46412-hf-tokenizer-fix-actually-fixed" \
    --cluster ai2/jupiter \
    --is_tuned \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --use_hf_tokenizer_template \
    --beaker_image "${BEAKER_IMAGE}" \
    --oe_eval_tasks "ifeval::hamish_zs_reasoning_deepseek" \
    --run_oe_eval_experiments \
    --evaluate_on_weka \
    --run_id placeholder \
    --oe_eval_max_length 32768 \
    --process_output r1_style \
    --skip_oi_evals

echo "Done!"
