# #!/bin/bash

python scripts/submit_eval_jobs.py \
  --model_name grpo_general_from_zero__1__1766650332_checkpoints_step50  \
  --location /weka/oe-adapt-default/allennlp/deletable_checkpoint/tengx/grpo_general_from_zero__1__1766650332_checkpoints/step_50 \
  --cluster   ai2/jupiter  \
  --evaluate_on_weka \
  --is_tuned \
  --workspace "ai2/tulu-3-results" \
  --priority normal \
  --preemptible \
  --use_hf_tokenizer_template \
  --run_oe_eval_experiments \
  --skip_oi_evals \
  --oe_eval_max_length 16384 \
  --step 250 \
  --gpu_multiplier 1 \
  --oe_eval_tasks alpaca_eval_v3::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,gpqa:0shot_cot::hamish_zs_reasoning_deepseek \
  --beaker_image oe-eval-beaker/oe_eval_auto

# aime:zs_cot_r1::pass_at_32_2024_temp0.6,aime:zs_cot_r1::pass_at_32_2025_temp0.6
  # --evaluate_on_weka \
# #!/bin/bash
# python scripts/submit_eval_jobs.py \
#   --model_name  sft_olmo2_lc_wildchat_8192_olmo2_lc_orz_mix_24175__1__1754343934_checkpoints\
#   --location gs://ai2-llm/post-training//tengx//output/sft_olmo2_lc_wildchat_8192_olmo2_lc_orz_mix_24175__1__1754343934_checkpoints/step_100 \
#   --cluster ai2/saturn-cirrascale ai2/neptune-cirrascale \
#   --is_tuned \
#   --workspace "tulu-3-results" \
#   --priority high \
#   --preemptible \
#   --use_hf_tokenizer_template \
#   --run_oe_eval_experiments \
#   --skip_oi_evals \
#   --oe_eval_max_length 8192 \
#   --step 250 \
#   --gpu_multiplier 2 \
#   --oe_eval_tasks mbppplus:0-shot-chat::tulu-thinker

