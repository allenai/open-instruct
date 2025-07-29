exp_name="grpo_orz_olmo3_fullmix"

# full integration mix
dataset_mix="saurabh5/rlvr_acecoder 56878 hamishivi/rlvr_orz_math_57k_collected 56878 hamishivi/tulu_3_rewritten_400k_string_f1_only_v2 56878 allenai/IF_multi_constraints_upto5 56878"
# math only mix
# dataset_mix="hamishivi/rlvr_orz_math_57k_collected 56878"

# all evals
# evals="minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,popqa::hamish_zs_reasoning,mmlu:cot::hamish_zs_reasoning,alpaca_eval_v3::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,mbppplus:0-shot-chat::tulu-thinker,codex_humanevalplus:0-shot-chat-v1::tulu-thinker"
# math evals
evals="minerva_math::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,aime::hamish_zs_reasoning"

# all I've changed with the checkpoints is the config.json, model_type=olmo3 and architectures is OLMo3ForCausalLM 
# jacob tulu sft
# model_name_or_path="/weka/oe-adapt-default/michaeln/olmo3/olmo3_reasoning-anneal-tulu3sft-olmo2hparams__8__1751523764/"
# midtraining no reasoning
# model_name_or_path="/weka/oe-adapt-default/michaeln/olmo3/anneal-round1-100B-olmo3_7b_no-reasoning-anneal-3c193128_step47684"
# midtraining with reasoning
# model_name_or_path="/weka/oe-adapt-default/michaeln/olmo3/anneal-round1-100B-olmo3_7b_with-reasoning-anneal-9d6f76b0_step47684"
# micro anneals
# model_name_or_path="/weka/oe-adapt-default/allennlp/deletable_checkpoint/michaeln/olmo3_microanneal-finemath-643cecc4_step4769-hf"
# model_name_or_path="/weka/oe-training-default/ai2-llm/checkpoints/OLMo3-midtraining/anneal-round1-100B-olmo3_7b_with-reasoning-anneal-12T-3d39e871/step47684-hf"
# model_name_or_path="/weka/oe-training-default/ai2-llm/checkpoints/kylel/baseline-olmo2_7b-928646-anneal-100B-dolma2-round1-alldressed-17b22b3a/step47684-hf"
# model_name_or_path="/weka/oe-training-default/ai2-llm/checkpoints/OLMo3-midtraining/anneal-round2-100B-olmo3_7b_with-reasoning-anneal-12T-53f443c7/step47684-hf"
model_name_or_path="/weka/oe-training-default/ai2-llm/checkpoints/OLMo3-midtraining/anneal-round3-webround2-100B-olmo3_7b_with-reasoning-anneal-12T-302b1ae8/step47684-hf"
gs_model_name="olmo3-midtraining-round3"
# cluster
cluster=ai2/augusta-google-1
# cluster=ai2/jupiter-cirrascale-2

python mason.py \
    --task_name ${exp_name} \
    --cluster ${cluster} \
    --workspace ai2/tulu-thinker \
    --priority high \
    --pure_docker_mode \
    --image michaeln/open_instruct_dev_uv_olmo3 \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --gs_model_name $gs_model_name \
    --budget ai2/oe-adapt \
    --gpus 8 -- \
source configs/beaker_configs/ray_node_setup.sh \&\& \
source configs/beaker_configs/code_api_setup.sh \&\& \
python open_instruct/grpo_fast.py \
    --exp_name ${exp_name} \
    --beta 0.0 \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 64 \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --kl_estimator kl3 \
    --dataset_mixer_list ${dataset_mix} \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/tulu_3_rewritten_100k 32 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 8192 \
    --max_prompt_token_length 2048 \
    --response_length 6144 \
    --pack_length 8192 \
    --model_name_or_path ${model_name_or_path} \
    --chat_template_name olmo_thinker_r1_style \
    --stop_strings "</answer>" \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 256000 \
    --deepspeed_stage 2 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --num_evals 5 \
    --save_freq 100 \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --mask_truncated_completions True \
    --oe_eval_max_length 8192 \
    --try_launch_beaker_eval_jobs_on_weka True \
    --oe_eval_tasks ${evals} \
    --oe_eval_beaker_image oe-eval-beaker/oe_eval_olmo3_auto
