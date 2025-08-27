# /weka/oe-adapt-default/allennlp/deletable_checkpoint/stellal/grpo_mathonly_1m_olmo3-meta-fvi-noisy-5b-5b-5b__1__1755969330_checkpoints
# 


steps=(
  100 200 300 400 500 600 700 800 900 1000 1200 1300
)
length=8192
for step in "${steps[@]}"; do

  echo "Running eval for $path"
  # model name to be the second to last part of the path
  model_name=$(basename $(dirname $path))_$(basename $path)
  echo "Model name: $model_name"
  python scripts/submit_eval_jobs.py \
        --model_name olmo3_10b_microanneal_metareasoning_13639_math_code_rl__1__1753343950_step_${step} \
        --location gs://ai2-llm/post-training//faezeb//output/olmo3_10b_microanneal_metareasoning_13639_math_code_rl__1__1753343950_checkpoints/step_${step} \
        --cluster ai2/augusta-google-1 \
        --workspace "ai2/tulu-3-results" \
        --priority normal \
        --preemptible \
        --run_oe_eval_experiments \
        --use_hf_tokenizer_template \
        --skip_oi_evals \
        --is_tuned \
        --run_id https://wandb.ai/ai2-llm/open_instruct_internal/runs/dhosov4z \
        --oe_eval_max_length $length \
        --step $step \
        --oe_eval_tasks minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,popqa::hamish_zs_reasoning,mmlu:cot::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,mbppplus:0-shot-chat::tulu-thinker,codex_humanevalplus:0-shot-chat-v1::tulu-thinker \
        --oe_eval_stop_sequences "</answer>" \
        --beaker_image oe-eval-beaker/oe_eval_olmo3_auto \
        --process_output r1_style
done 