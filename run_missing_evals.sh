#!/usr/bin/env bash
set -euo pipefail

# code base (run 1) — regular tasks (1 missing)

uv run python scripts/submit_eval_jobs.py \
    --model_name flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B \
    --location /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B/step620-hf \
    --cluster ai2/saturn ai2/ceres --is_tuned --workspace ai2/flex2 --priority urgent --preemptible --use_hf_tokenizer_template --run_oe_eval_experiments --evaluate_on_weka --run_id placeholder --oe_eval_max_length 4096 --process_output r1_style --skip_oi_evals \
    --oe_eval_tasks ifeval_ood::tulu-thinker \
    --beaker_image jacobm/oe-eval-flex-olmo-9-29-5
