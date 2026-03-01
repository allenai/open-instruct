#!/bin/bash
# 2 extra omega full runs for Instruct SFT model

set -e

BEAKER_IMAGE="yanhongl/oe_eval_olmo3_devel_v7"
HF_LOCATION="allenai/OLMo-3.2-Hybrid-7B-Instruct-SFT"

for N in 4 5; do
    echo "-> Omega run ${N}"
    uv run scripts/submit_eval_jobs.py \
        --model_name "hf-OLMo-3.2-Hybrid-7B-Instruct-SFT-omega-${N}" \
        --location "${HF_LOCATION}" \
        --cluster ai2/jupiter \
        --is_tuned \
        --workspace ai2/olmo-instruct \
        --priority urgent \
        --gpu_multiplier 2 \
        --preemptible \
        --use_hf_tokenizer_template \
        --beaker_image "${BEAKER_IMAGE}" \
        --oe_eval_tasks "omega:0-shot-chat_deepseek" \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 32768 \
        --process_output r1_style \
        --skip_oi_evals
done

echo "Done!"
