# full integration mix
# dataset_mix="saurabh5/rlvr_acecoder_filtered 63033 hamishivi/rlvr_orz_math_57k_collected 56878 hamishivi/tulu_3_rewritten_400k_string_f1_only_v2 56878 allenai/IF_multi_constraints_upto5 56878"
# math only mix
dataset_mix="mnoukhov/DAPO-Math-14k-Processed-RLVR 1.0"
# dataset_mix="hamishivi/hamishivi_rlvr_orz_math_57k_collected_all_filtered_hamishivi_qwen2_5_openthoughts2 1.0"

# all evals
# evals="minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,popqa::hamish_zs_reasoning,mmlu:cot::hamish_zs_reasoning,alpaca_eval_v3::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,mbppplus:0-shot-chat::tulu-thinker,codex_humanevalplus:0-shot-chat-v1::tulu-thinker"
# math evals
evals="minerva_math::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,aime:zs_cot_r1::pass_at_32_2024_temp1,aime:zs_cot_r1::pass_at_32_2025_temp1"

# all I've changed with the checkpoints is the config.json, model_type=olmo3 and architectures is OLMo3ForCausalLM 
# model_name_or_path="/weka/oe-training-default/ai2-llm/checkpoints/OLMo3-midtraining/anneal-round5-100B-olmo25_7b-anneal-2T-f07e3111/step47684-hf"
# gs_model_name="olmo2.5-2T-midtraining-round5-100B"

# model_name_or_path="gs://ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_6T_M100B_r5-midtrain_round3_qwenlike_s2pdf_gzip2080_just-synth-cwe-yake_yarn-fullonly_10B-c6bda7ae/step2385-hf"
model_name_or_path="/weka/oe-training-default/ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_6T_M100B_r5-midtrain_round3_qwenlike_s2pdf_gzip2080_just-synth-cwe-yake_yarn-fullonly_10B-c6bda7ae/step2385-hf"
gs_model_name="olmo2.5-R5-10BLC-c6bda7ae"

exp_name="grpo_dapo14k_${gs_model_name}"
EXP_NAME=${EXP_NAME:-${exp_name}}


# cluster
cluster=ai2/augusta-google-1
# cluster=ai2/jupiter-cirrascale-2

NUM_NODES=4

python mason.py \
    --task_name ${EXP_NAME} \
    --cluster ${cluster} \
    --workspace ai2/tulu-thinker \
    --priority high \
    --pure_docker_mode \
    --image michaeln/open_instruct_olmo2_retrofit \
    --preemptible \
    --num_nodes $NUM_NODES \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND="FLASH_ATTN" \
    --gs_model_name $gs_model_name \
   --gpus 8 \
    --budget ai2/oe-adapt \
    -- \
source configs/beaker_configs/ray_node_setup.sh \&\& \
source configs/beaker_configs/code_api_setup.sh \&\& \
python open_instruct/grpo_fast.py \
    --exp_name ${EXP_NAME} \
    --beta 0.0 \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 128 \
    --num_mini_batches 4 \
    --num_epochs 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --kl_estimator kl3 \
    --dataset_mixer_list ${dataset_mix} \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/tulu_3_rewritten_100k 32 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 18432 \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 32768 \
    --model_name_or_path ${model_name_or_path} \
    --chat_template_name olmo_simple_thinker_r1_style \
    --stop_strings "</answer>" \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 1024000 \
    --deepspeed_stage 3 \
    --num_learners_per_node 8 \
    --vllm_num_engines 24 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 25 \
    --save_freq 25 \
    --checkpoint_state_freq 25 \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --mask_truncated_completions True \
    --oe_eval_max_length 8192 \
    --try_launch_beaker_eval_jobs_on_weka True \
    --oe_eval_tasks ${evals} \
    --oe_eval_beaker_image oe-eval-beaker/oe_eval_olmo2_retrofit_auto
