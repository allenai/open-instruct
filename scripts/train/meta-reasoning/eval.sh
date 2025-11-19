# /weka/oe-adapt-default/allennlp/deletable_checkpoint/stellal/grpo_mathonly_1m_olmo3-meta-fvi-noisy-5b-5b-5b__1__1755969330_checkpoints
# /weka/oe-adapt-default/allennlp/deletable_checkpoint/stellal/grpo_mathonly_1m_olmo3-meta-noisy-fvi-5b-5b-5b__1__1755969488_checkpoints

# /weka/oe-adapt-default/allennlp/deletable_checkpoint/stellal/grpo_spurious_dapo_nochat_olmo2.5-6T-LC_midtrain_round3__1__1757890448_checkpoints
# /weka/oe-adapt-default/allennlp/deletable_checkpoint/stellal/grpo_spurious_1m_olmo2.5-6T-LC_midtrain_round3_with_yarn__1__1757724902_checkpoints/

# base_dir=/weka/oe-adapt-default/allennlp/deletable_checkpoint/stellal

# model_id=grpo_spurious_dapo_nochat_olmo2.5-6T-LC_midtrain_round3__1__1757890448
# wandb_id=6ha9jg24

# # model_id=grpo_spurious_1m_olmo2.5-6T-LC_midtrain_round3_with_yarn__1__1757724902
# # wandb_id=6yvj7es6

# steps=(
#   25 50 100 150 200 250 300
# )
# length=8192
# for step in "${steps[@]}"; do

#   echo "Running eval for $path"
#   # model name to be the second to last part of the path
#   # model_name=$(basename $(dirname $path))_$(basename $path)
#   model_name="${model_id}_step_${step}"
#   echo "Model name: $model_name"
#   python scripts/submit_eval_jobs.py \
#         --model_name ${model_id}_step_${step} \
#         --location ${base_dir}/${model_id}_checkpoints/step_${step} \
#         --cluster ai2/neptune \
#         --workspace "ai2/tulu-3-results" \
#         --priority high \
#         --preemptible \
#         --run_oe_eval_experiments \
#         --use_hf_tokenizer_template \
#         --skip_oi_evals \
#         --is_tuned \
#         --run_id https://wandb.ai/ai2-llm/open_instruct_internal/runs/${wandb_id} \
#         --oe_eval_max_length $length \
#         --step $step \
#         --oe_eval_tasks gsm8k::zs_cot_latex,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,popqa::hamish_zs_reasoning,mmlu:cot::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,mbppplus:0-shot-chat::tulu-thinker,codex_humanevalplus:0-shot-chat-v1::tulu-thinker \
#         --oe_eval_stop_sequences "</answer>" \
#         --beaker_image oe-eval-beaker/oe_eval_olmo2_retrofit_auto \
#         --process_output r1_style
# done

NEXT_MODEL_DEV=(
    # Knowledge
    "mmlu:cot::hamish_zs_reasoning_deepseek"
    # "popqa::hamish_zs_reasoning_deepseek"
    # "simpleqa::tulu-thinker_deepseek"
    
    # Reasoning
    # "bbh:cot::hamish_zs_reasoning_deepseek_v2" # OLD: "bbh:cot::hamish_zs_reasoning_deepseek"
    "gpqa:0shot_cot::qwen3-instruct"
    "zebralogic::hamish_zs_reasoning_deepseek"
    # "agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek"

    # Math
    # [faster] minerva_math_500::hamish_zs_reasoning
    "minerva_math::hamish_zs_reasoning_deepseek"
    "gsm8k::zs_cot_latex_deepseek"
    "omega_500:0-shot-chat_deepseek" # OLD: "omega:0-shot-chat"
    "aime:zs_cot_r1::pass_at_32_2024_deepseek"
    "aime:zs_cot_r1::pass_at_32_2025_deepseek"  # OLD: "aime::hamish_zs_reasoning"
    
    # Coding
    "codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek"
    "mbppplus:0-shot-chat::tulu-thinker_deepseek"
    # "livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags"
    # [TODO not merged] codeeditorbench - requires separate server
    # [TODO, maybe] cruxeval
    
    # Chat / IF / Vibes
    "alpaca_eval_v3::hamish_zs_reasoning_deepseek"
    "ifeval::hamish_zs_reasoning_deepseek"
    # [expensive, multi-turn all versions] multiturn_alpacaeval::tulu
    # [expensive, typos vibes] styled_evals::tulu
    # [optional, typos compare] styled_math500::tulu
    # [optional, typos compare] styled_popqa::tulu
    # [optional, typos compare] styled_truthfulqa::tulu

    # Tool Use
    # "bfcl_all::std" # This requires special logic on model_args and metadata, handled below
)
# mmlu:cot::hamish_zs_reasoning_deepseek,gpqa:0shot_cot::qwen3-instruct,zebralogic::hamish_zs_reasoning_deepseek,minerva_math::hamish_zs_reasoning_deepseek,gsm8k::zs_cot_latex_deepseek,omega_500:0-shot-chat_deepseek,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek


base_dir=/weka/oe-adapt-default/allennlp/deletable_checkpoint/hamishivi

model_id=grpo_spurious_1m_olmo3-LC_base_final__1__1761681197
wandb_id=vi78czn1

steps=(
  25 50 75 100 125 150 175
)
length=8192
for step in "${steps[@]}"; do

  echo "Running eval for $path"
  # model name to be the second to last part of the path
  # model_name=$(basename $(dirname $path))_$(basename $path)
  model_name="${model_id}_step_${step}"
  echo "Model name: $model_name"
  python scripts/submit_eval_jobs.py \
        --model_name ${model_id}_step_${step} \
        --location ${base_dir}/${model_id}_checkpoints/step_${step} \
        --cluster ai2/neptune \
        --workspace "ai2/tulu-3-results" \
        --priority urgent \
        --preemptible \
        --run_oe_eval_experiments \
        --use_hf_tokenizer_template \
        --skip_oi_evals \
        --is_tuned \
        --run_id https://wandb.ai/ai2-llm/open_instruct_internal/runs/${wandb_id} \
        --oe_eval_max_length $length \
        --step $step \
        --oe_eval_tasks mmlu:cot::hamish_zs_reasoning_deepseek,gpqa:0shot_cot::qwen3-instruct,zebralogic::hamish_zs_reasoning_deepseek,minerva_math::hamish_zs_reasoning_deepseek,gsm8k::zs_cot_latex_deepseek,omega_500:0-shot-chat_deepseek,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek \
        --oe_eval_stop_sequences "</answer>" \
        --beaker_image oe-eval-beaker/oe_eval_olmo2_retrofit_auto \
        --process_output r1_style
done



# "mmlu:cot::hamish_zs_reasoning_deepseek" "popqa::hamish_zs_reasoning_deepseek" "simpleqa::tulu-thinker_deepseek" "bbh:cot::hamish_zs_reasoning_deepseek" "gpqa:0shot_cot::hamish_zs_reasoning_deepseek" "zebralogic::hamish_zs_reasoning_deepseek" "agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek" "minerva_math::hamish_zs_reasoning_deepseek" "gsm8k::zs_cot_latex_deepseek" "omega_500:0-shot-chat_deepseek" "aime:zs_cot_r1::pass_at_32_2024_deepseek" "aime:zs_cot_r1::pass_at_32_2025_deepseek" "codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek" "mbppplus:0-shot-chat::tulu-thinker_deepseek" "livecodebench_codegeneration::tulu-thinker_deepseek" "alpaca_eval_v3::hamish_zs_reasoning_deepseek" "ifeval::hamish_zs_reasoning_deepseek"


# model_ids=(
#   # grpo_mathonly_1m_olmo2.5-2T-everything-r19-random__1__1756580755
#   # grpo_mathonly_1m_olmo2.5-2T-mathcodereasoning-r20-random__1__1756606736
#   # grpo_mathonly_1m_olmo2.5-2T-superswarm-noreasoning-r18-random__1__1756765818
#   grpo_spurious_1m_olmo2.5-6T-LC_midtrain_round3_with_yarn__1__1757461424
# )
# run_id=tk9torhz

# for model_id in "${model_ids[@]}"; do
#   location=/weka/oe-adapt-default/allennlp/deletable_checkpoint/stellal/${model_id}_checkpoints

#   steps=(
#     50 100 150 200 250 300 350 400 450 500
#   )

#   length=8192

#   for step in "${steps[@]}"; do

#     echo "Running eval for $path"
#     # model name to be the second to last part of the path
#     # model_name=$(basename $(dirname $path))_$(basename $path)
#     # echo "Model name: $model_name"
#     python scripts/submit_eval_jobs.py \
#           --model_name ${model_id}_step_${step} \
#           --location ${location}/step_${step} \
#           --cluster ai2/jupiter-cirrascale-2 \
#           --workspace "ai2/tulu-3-results" \
#           --priority high \
#           --preemptible \
#           --run_oe_eval_experiments \
#           --use_hf_tokenizer_template \
#           --skip_oi_evals \
#           --is_tuned \
#           --run_id https://wandb.ai/ai2-llm/open_instruct_internal/runs/${run_id} \
#           --oe_eval_max_length $length \
#           --step $step \
#           --oe_eval_tasks minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,popqa::hamish_zs_reasoning,mmlu:cot::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,mbppplus:0-shot-chat::tulu-thinker,codex_humanevalplus:0-shot-chat-v1::tulu-thinker \
#           --oe_eval_stop_sequences "</answer>" \
#           --beaker_image oe-eval-beaker/oe_eval_olmo2_retrofit_auto \
#           --process_output r1_style

#   done
# done 