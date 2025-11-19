
# general mix


# hamishivi/IF_multi_constraints_upto5_filtered
# hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2_filtered
# hamishivi/virtuoussy_multi_subject_rlvr_filtered
# hamishivi/new-wildchat-english-general_filtered
# hamishivi/diverse-semi-verifiable-tasks-o3-7500-o4-mini-high_filtered
# hamishivi/All_Puzzles_filtered
# hamishivi/open-code-reasoning-rlvr-stdio_filtered
# hamishivi/klear-code-rlvr_filtered
# hamishivi/synthetic2-rlvr-code-compressed_filtered
# hamishivi/llama-nemotron-rlvr-difficulty-6_filtered
# hamishivi/llama-nemotron-rlvr-difficulty-7_filtered
# hamishivi/llama-nemotron-rlvr-difficulty-8_filtered
# hamishivi/llama-nemotron-rlvr-difficulty-9_filtered
# hamishivi/llama-nemotron-rlvr-difficulty-10_filtered
# hamishivi/rlvr_acecoder_filtered_filtered
# hamishivi/MathSub-30K_filtered
# hamishivi/rlvr_orz_math_57k_collected_filtered
# hamishivi/omega-combined-no-boxed_filtered
# hamishivi/DAPO-Math-17k-Processed_filtered
# hamishivi/AceReason-Math_filtered
# hamishivi/deepscaler_20k_medhard_nolatex_rlvr_filtered

# allenai/new-wildchat-english-general 19537
dataset_general_mix="hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2 25000 hamishivi/virtuoussy_multi_subject_rlvr 40000 allenai/new-wildchat-english-general 19000"
dataset_general_mix_new="allenai/new-wildchat-english-general 1.0 faezeb/diverse-semi-verifiable-tasks-o3-7500-o4-mini-high 1.0"
dataset_general_mix_all="hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2 25000 hamishivi/virtuoussy_multi_subject_rlvr 40000 allenai/new-wildchat-english-general 19000 faezeb/diverse-semi-verifiable-tasks-o3-7500-o4-mini-high 10000"

nonreasoner_integration_mix="saurabh5/rlvr_acecoder_filtered 20000 hamishivi/omega-combined 20000 hamishivi/rlvr_orz_math_57k_collected 14000 hamishivi/polaris_53k 14000 TTTXXX01/MathSub-30K 9000 hamishivi/DAPO-Math-17k-Processed 7000 allenai/IF_multi_constraints_upto5 35000 hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2 22000 hamishivi/virtuoussy_multi_subject_rlvr 20000 allenai/new-wildchat-english-general 19000" # 192k # TTTXXX01/All_Puzzles 14000 # faezeb/diverse-semi-verifiable-tasks-o3-7500-o4-mini-high 10000
nonreasoner_integration_mix_decon="hamishivi/rlvr_acecoder_filtered_filtered 20000 hamishivi/omega-combined-no-boxed_filtered 20000 hamishivi/rlvr_orz_math_57k_collected_filtered 14000 hamishivi/polaris_53k 14000 hamishivi/MathSub-30K_filtered 9000 hamishivi/DAPO-Math-17k-Processed_filtered 7000 allenai/IF_multi_constraints_upto5_filtered_dpo_0625_filter-keyword-filtered-topic-char-topic-filtered 38000 allenai/rlvr_general_mix-keyword-filtered-topic-chars-char-filt-topic-filtered 50000" #hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2_filtered 22000 hamishivi/virtuoussy_multi_subject_rlvr_filtered 20000 hamishivi/new-wildchat-english-general_filtered 19000"
nonreasoner_no_math_mix_decon="hamishivi/rlvr_acecoder_filtered_filtered 20000 allenai/IF_multi_constraints_upto5_filtered_dpo_0625_filter-keyword-filtered-topic-char-topic-filtered 38000 allenai/rlvr_general_mix-keyword-filtered-topic-chars-char-filt-topic-filtered 50000" # hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2_filtered 22000 hamishivi/new-wildchat-english-general_filtered 19000 hamishivi/virtuoussy_multi_subject_rlvr_filtered 20000 (old not luca filtered)
nonreasoner_math_mix_decon="hamishivi/omega-combined-no-boxed_filtered 20000 hamishivi/rlvr_orz_math_57k_collected_filtered 14000 hamishivi/polaris_53k 14000 hamishivi/MathSub-30K_filtered 9000 hamishivi/DAPO-Math-17k-Processed_filtered 7000"


echo "dataset_general_mix: $nonreasoner_no_math_mix_decon"

# all evals
evals="minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,popqa::hamish_zs_reasoning,mmlu:cot::hamish_zs_reasoning,alpaca_eval_v3::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,mbppplus:0-shot-chat::tulu-thinker,codex_humanevalplus:0-shot-chat-v1::tulu-thinker"
# math evals
math_evals="minerva_math::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,aime:zs_cot_r1::pass_at_32_2024_temp1,aime:zs_cot_r1::pass_at_32_2025_temp1"
general_evals="minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,popqa::hamish_zs_reasoning,mmlu:cot::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,mbppplus:0-shot-chat::tulu-thinker,codex_humanevalplus:0-shot-chat::tulu-thinker,alpaca_eval_v3::hamish_zs_reasoning,aime:zs_cot_r1::pass_at_32_2024_temp1,aime:zs_cot_r1::pass_at_32_2025_temp1,omega_500:0-shot-chat"
general_eval_ds="minerva_math::hamish_zs_reasoning_deepseek,gsm8k::zs_cot_latex_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,zebralogic::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,gpqa:0shot_cot::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,mmlu:cot::hamish_zs_reasoning_deepseek,bbh:cot::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,omega_500:0-shot-chat_deepseek,simpleqa::tulu-thinker_deepseek"
general_evals_int="gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,omega_500:0-shot-chat_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,bbh:cot::hamish_zs_reasoning_deepseek_v2,mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek"


weka_model_name="olmo3-reason-nonreason-sft-lc-dpo-delta125k"
gs_model_name="olmo3-r-nonreason-sft-lc-dpo-delta125k-rl350"
gs_model_name="olmo3-r-nonreason-sft-lc-permissive-dpo"
gs_model_name="olmo3-instruct-dpo-hpz1"
# gs_model_name="olmo3-r-nonreason-permissive-dpo-rl250"
model_name_or_path="/weka/oe-adapt-default/scottg/olmo/merging/ckpts/smR3-0926-dpo-delta125k_vgraf-gpt125k_y1-1e-6__42__1759101621"
model_name_or_path="gs://ai2-llm/post-training//faezeb/output/grpo_int_mix_p32_4_olmo3-reason-nonreason-sft-lc-dpo-delta125k__1__1759770678_checkpoints/step_350"
model_name_or_path="/weka/oe-adapt-default/allennlp/deletable_checkpoint/faezeb/final_olmo_instruct_rl/grpo_int_mix_p32_4_olmo3-reason-nonreason-sft-lc-dpo-delta125k__1__1759770678_checkpoints/step_350" # step 350 of previous RL run with pass rate reward for code
model_name_or_path="/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/final-checkpoints/permissive-lr-8e-5-seed11235-YOLO-RUN-DPO"
model_name_or_path="/weka/oe-adapt-default/allennlp/deletable_checkpoint/scottg/olmo3-instruct-final-dpo-lbc100-s64-125k-lr1e-6__64__1762642896" # just a test unhinged final dpo nov 14th
model_name_or_path="/weka/oe-adapt-default/allennlp/deletable_checkpoint/victoriag/olmo3-7b-DPO-1115-newbase-1e-6__42__1763265204" # nov 15 test checkpoint
# model_name_or_path="/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-instruct-dpo-1116-vibes/olmo3-7b-DPO-1115-newb-tpc-d5-lbc100-bal-1e-6-1__42__1763293644" # nov 16 tentative final checkpoint row 33 # was replace by row 27
model_name_or_path="/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-instruct-dpo-1116-vibes/olmo3-7b-DPO-1115-newbase-tpc-dedup5-1e-6-hpz1__42__1763287884" # nov 16 tentative final checkpoint row 21
# model_name_or_path="/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-instruct-dpo-1116-vibes/olmo3-7b-DPO-1115-newb-tpc-d5-lbc100-1e-6-hpz1__42__1763329758" # nov 16 tentative final checkpoint row 27 lbc
# model_name_or_path="/weka/oe-adapt-default/allennlp/deletable_checkpoint/faezeb/FINAL_olmo_instruct_7b_rl/grpo_math_only_p64_4_8k_olmo3-instruct-dpo-lbc__1__1763362445_checkpoints/step_200"


# gs_model_name="olmo2.5-6T-lc-r1-reasoning"

# model_name_or_path="/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/olmo3-hparam-search/olmo2.5-LC-R3-olmo2-tulu3-mix-num_3"
# gs_model_name="olmo2.5-lc-r3-jacobsft-mix3"


# exp_name="grpo_int_mix_p64_4_${weka_model_name}"
# exp_name="grpo_nonreasoner_gnrlmix_new_${weka_model_name}2"
# exp_name="grpo_nonreasoner_reasonmix_${weka_model_name}2"
# exp_name="grpo_nonreasoner_gnrlmix_${weka_model_name}"
# exp_name="grpo_reasoner_gnrlmix_all_${weka_model_name}"
# EXP_NAME=${EXP_NAME:-${exp_name}}


# cluster
cluster=ai2/augusta #ai2/augusta-google-1
# cluster=ai2/jupiter-cirrascale-2
chat_template=olmo123 #olmo

NUM_GPUS=${NUM_GPUS:-8}
# stella image: stellal/open_instruct_dev
# michael's image:  michaeln/open_instruct_olmo2_retrofit
# my image: faezeb/open_instruct_olmo25
# vllm servers:
# saturn-cs-aus-244.reviz.ai2.in:8002  (qwen3-32b) x4
# saturn-cs-aus-254.reviz.ai2.in:8001  (qwen3-32b)
# saturn-cs-aus-245.reviz.ai2.in:8003  (qwen3-32b) x4

# ai2/tulu-thinker
num_unique_prompts=(
    64 #64 Michael's suggestion to try smaller prompt set
)

num_mini_batches=(
    4
)

# --env VLLM_ATTENTION_BACKEND="FLASH_ATTN" \ ### hamish suggested too remove

for num_unique_prompt in "${num_unique_prompts[@]}"; do

    if [ $num_unique_prompt -eq 32 ]; then
        # hosted_vllm=http://saturn-cs-aus-245.reviz.ai2.in:8003/v1
        hosted_vllm="http://saturn-cs-aus-250.reviz.ai2.in:8001/v1"
        exp_name="grpo_int_mix_p32_4_${weka_model_name}"
    else

        # hosted_vllm="http://ceres-cs-aus-451.reviz.ai2.in:8003/v1" # died
        # hosted_vllm="http://saturn-cs-aus-250.reviz.ai2.in:8001/v1" # seems not to be working
        # hosted_vllm="http://saturn-cs-aus-243.reviz.ai2.in:8004/v1" # urgent / working
        # hosted_vllm="http://saturn-cs-aus-237.reviz.ai2.in:8002/v1" # working
        hosted_vllm="http://ceres-cs-aus-446.reviz.ai2.in:8003/v1"
        exp_name="grpo_math_only_p64_4_8k_${gs_model_name}"
    fi

    # BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
    # BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

    # --image "saurabhs/open-instruct-integration-test-main" \
    # --image "01K9ZKASZV7KWMB1D1RWGDN18C" # finbarr
    # --image "hamishivi/open_instruct_rl32_no_ref19" # hamish image with rl32b vllm
    # --image faezeb/open-instruct-integration-test-fae-new-rl

    EXP_NAME=${EXP_NAME:-${exp_name}}
    for num_mini_batch in "${num_mini_batches[@]}"; do
        uv run python mason.py \
                --description $exp_name \
                --task_name ${EXP_NAME} \
                --cluster ${cluster} \
                --workspace ai2/olmo-instruct  \
                --priority urgent \
                --pure_docker_mode \
                --image hamishivi/open_instruct_rl32_no_ref19 \
                --preemptible \
                --num_nodes 8 \
                --max_retries 5 \
                --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
                --env HOSTED_VLLM_API_BASE=$hosted_vllm \
                --gs_model_name $gs_model_name \
                --gpus ${NUM_GPUS} \
                --budget ai2/oe-adapt -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
                --exp_name ${EXP_NAME} \
                --beta 0.0 \
                --num_samples_per_prompt_rollout 8 \
                --num_unique_prompts_rollout $num_unique_prompt \
                --num_mini_batches $num_mini_batch \
                --num_epochs 1 \
                --learning_rate 1e-6 \
                --per_device_train_batch_size 1 \
                --kl_estimator kl3 \
                --dataset_mixer_list ${nonreasoner_math_mix_decon} \
                --dataset_mixer_list_splits train \
                --dataset_mixer_eval_list hamishivi/omega-combined 4 allenai/IF_multi_constraints_upto5 4 saurabh5/rlvr_acecoder_filtered 4 hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2 4 hamishivi/virtuoussy_multi_subject_rlvr 4 \
                --dataset_mixer_eval_list_splits train \
                --max_prompt_token_length 2048 --response_length 8192 --pack_length 11264 \
                --model_name_or_path ${model_name_or_path} \
                --chat_template_name ${chat_template} \
                --stop_strings "</answer>" \
                --non_stop_penalty False \
                --temperature 1.0 \
                --total_episodes 1024000 \
                --deepspeed_stage 3 \
                --num_learners_per_node 8 \
                --vllm_num_engines 56 \
                --lr_scheduler_type constant \
                --apply_verifiable_reward true \
                --seed 1 \
                --local_eval_every 50 \
                --save_freq 50 \
                --checkpoint_state_freq 50 \
                --beaker_eval_freq 50 \
                --gradient_checkpointing \
                --with_tracking \
                --vllm_enable_prefix_caching \
                --clip_higher 0.272 \
                --mask_truncated_completions False \
                --llm_judge_model hosted_vllm/Qwen/Qwen3-32B \
                --llm_judge_timeout 600 \
                --llm_judge_max_tokens 2048 \
                --llm_judge_max_context_length 32768 \
                --oe_eval_max_length 32768 \
                --try_launch_beaker_eval_jobs_on_weka True \
                --oe_eval_tasks ${general_evals_int} \
                --eval_priority urgent \
                --code_pass_rate_reward_threshold 0.99 \
                --inflight_updates true \
                --async_steps 8 \
                --active_sampling \
                --advantage_normalization_type centered \
                --no_resampling_pass_rate 0.875 \
                # --oe_eval_beaker_image oe-eval-beaker/oe_eval_olmo2_retrofit_auto $@
    done
done

                # --code_pass_rate_reward_threshold 0.99 \

# 16k length settings
                # --max_token_length 18432 \
                # --max_prompt_token_length 2048 \
                # --response_length 16384 \
                # --pack_length 18432 \


# for non-reasoner sft sc
        # --max_token_length 8192 \
        # --max_prompt_token_length 2048 \
        # --response_length 6144 \
        # --pack_length 8192 \

# sc
        # --num_learners_per_node 8 \
        # --vllm_num_engines 8 \

        