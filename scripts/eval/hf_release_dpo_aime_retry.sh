#!/bin/bash
# 3 extra AIME 2024 runs for DPO model

set -e

BEAKER_IMAGE="yanhongl/oe_eval_olmo3_devel_v7"
HF_LOCATION="allenai/OLMo-3.2-Hybrid-7B-DPO"

for N in 4 5 6; do
    echo "-> AIME 2024 run ${N}"
    uv run scripts/submit_eval_jobs.py \
        --model_name "hf-OLMo-3.2-Hybrid-7B-DPO-aime-${N}" \
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
done

echo "Done!"
