
# write a for loop to run the eval script steps 50 100 150 200 and 250
#50 100 150 200 250

path="/weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_final_anneal/step11921-hf"
path="/weka/oe-adapt-default/faezeb/model_checkpoints/olmo2-cl-7b/step11921-hf"
path="/weka/oe-adapt-default/faezeb/model_checkpoints/olmo2-7b_10b-anneal_web-math-reasoning-a5c1c043/step4769-hf"
weka_path="/weka/oe-adapt-default/hamishi/model_checkpoints/rl_vs_sft/2605rl_base_ovlntrue_split_math_only_21744_step_700/2605rl_base_ovlntrue_split_math_only_21744__1__1748295856_checkpoints/step_700"
# get run id as the last part of the path
random_id=$(basename $path)

# if model has "olmo" in the name use max length 4096 otherwise use 32768
if [[ $path == *"olmo2"* ]]; then
    length=4096
elif [[ $path == *"olmo3"* ]]; then
    length=8192
else
    length=32768
fi




# olmo2 rl'ed on top of sft'ed olmo25 non-reasoner
general_eval_ds="minerva_math::hamish_zs_reasoning_deepseek,gsm8k::zs_cot_latex_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,zebralogic::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,gpqa:0shot_cot::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,mmlu:cot::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,omega_500:0-shot-chat_deepseek,simpleqa::tulu-thinker_deepseek,bbh:cot::hamish_zs_reasoning_deepseek_v2"
general_evals_int="gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,omega_500:0-shot-chat_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,bbh:cot::hamish_zs_reasoning_deepseek_v2,mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek"



# steps=(
#   50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950
# )
steps=(
  400 500 550
)

cluster="ai2/augusta"
other_clusters="ai2/saturn ai2/neptune ai2/jupiter"
# path="/weka/oe-adapt-default/allennlp/deletable_checkpoint/faezeb/grpo_int_mix_olmo2.5-6T-nonreasoner-sft-lc__1__1757453714_checkpoints"
path="gs://ai2-llm/post-training//faezeb/output/grpo_gnrlmix_all_advcenter_olmo2.5-6T-reasoner-sft__1__1757606629_checkpoints"
# path="gs://ai2-llm/post-training//faezeb/output/grpo_int_mix_olmo2.5-6T-nonreasoner-sft-lc-dpo__1__1757607393_checkpoints"
path="gs://ai2-llm/post-training//faezeb/output/grpo_int_mix_olmo2.5-6T-reason-nonreasoner-sft-lc-dpo__1__1757626784_checkpoints"
#"gs://ai2-llm/post-training//faezeb/output/grpo_gnrlmix_all_olmo2.5-6T-reasoner-sft__1__1757456313_checkpoints"
#"/weka/oe-adapt-default/allennlp/deletable_checkpoint/faezeb/grpo_int_mix_olmo2.5-6T-nonreasoner-sft-lc__1__1757453714_checkpoints"
#
#"/weka/oe-adapt-default/allennlp/deletable_checkpoint/faezeb/grpo_int_mix_olmo2.5-6T-nonreasoner-sft-lc__1__1757453714_checkpoints"
#"/weka/oe-adapt-default/allennlp/deletable_checkpoint/faezeb/grpo_nonreasoner_gnrlmix_olmo2.5-6T-sft-dpo__1__1757119639_checkpoints"

### FINAL INSTRUCT CHECKPOINTS
path="gs://ai2-llm/post-training//faezeb/output/grpo_math_mix8k_p64_4_F_olmo3-r-nonreason-sft-lc-permissive-dpo__1__1760644496_checkpoints"
# path="gs://ai2-llm/post-training//faezeb/output/grpo_math_8k_p64_4_F_seq_olmo3-r-nonreason-permissive-dpo-rl250__1__1760734702_checkpoints"
# path="/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/final-checkpoints/permissive-lr-8e-5-seed11235-YOLO-RUN-DPO"
path="gs://ai2-llm/post-training//faezeb/output/grpo_all_mix_p64_4_8k_olmo3-r-nonreason-sft-lc-permissive-dpo__1__1760977564_checkpoints"
path="/weka/oe-adapt-default/tongc/model_checkpoints/olmo3-mixed-grpo-fact-beta1em2-lr5em6/olmo3-mixed-grpo-fact-beta1em2-lr5em6__1__1761866622_checkpoints"
path="gs://ai2-llm/post-training//faezeb/output/grpo_all_mix_p64_4_8k_olmo3-instruct-test-final-dpo-v2__1__1763177452_checkpoints"


# if nonreasoner in path then use 16384 else if reasoner in path use 32768
# if [[ $path == *"nonreasoner"* ]]; then
#     length=16384
# elif [[ $path == *"reasoner"* ]]; then
#     length=32768
# else
#     length=8192
# fi
length=32768

for step in "${steps[@]}"; do

  echo "Running eval for ${path}/step_${step}"
  # model name to be last part of the path, ommitting the suffix `_checkpoints` from the end
  model_name=$(basename $path | sed 's/_checkpoints//')_step_${step}
  echo "Model name: $model_name"
  echo "Evaluate at length: $length"
  python scripts/submit_eval_jobs.py \
        --model_name ${model_name} \
        --location ${path}/step_${step} \
        --workspace "ai2/tulu-3-results" \
        --priority urgent \
        --preemptible \
        --cluster ${other_clusters} \
        --run_oe_eval_experiments \
        --use_hf_tokenizer_template \
        --skip_oi_evals \
        --is_tuned \
        --oe_eval_max_length ${length} \
        --oe_eval_tasks ${general_evals_int} \
        --oe_eval_stop_sequences "</answer>" \
        --beaker_image oe-eval-beaker/oe_eval_auto \
        --step $step \

        # --beaker_image oe-eval-beaker/oe_eval_olmo2_retrofit_auto \


        # --oe_eval_tasks agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek \
        # --oe_eval_tasks $general_evals \
        # --oe_eval_tasks simpleqa::tulu-thinker,livecodebench_codegeneration::tulu-thinker \
        # --evaluate_on_weka \
        # --oe_eval_tasks aime:zs_cot_r1::pass_at_32_2024_temp1,aime:zs_cot_r1::pass_at_32_2025_temp1,omega_500:0-shot-chat,livecodebench_codegeneration::tulu-thinker,simpleqa::tulu-thinker \
        # --oe_eval_tasks aime:zs_cot_r1::pass_at_32_2024_temp1,aime:zs_cot_r1::pass_at_32_2025_temp1,omega_500:0-shot-chat,livecodebench_codegeneration::tulu-thinker,simpleqa::tulu-thinker \
        # --evaluate_on_weka \
        # --oe_eval_tasks aime:zs_cot_r1::pass_at_32_2024_temp1,aime:zs_cot_r1::pass_at_32_2025_temp1,omega_500:0-shot-chat
        # --process_output 
        # --evaluate_on_weka \
                # --run_id https://wandb.ai/ai2-llm/open_instruct_internal/runs/dhosov4z \
done 
