exp_name="grpo_fast_olmo3_test"
model_name_or_path="/weka/oe-adapt-default/allennlp/deletable_checkpoint/michaeln/olmo3_reasoning-anneal-tulu3sft-olmo2hparams__8__1751523764/"
dataset_mix="saurabh5/rlvr_acecoder 56878 hamishivi/rlvr_orz_math_57k_collected 56878 hamishivi/tulu_3_rewritten_400k_string_f1_only_v2 56878 allenai/IF_multi_constraints_upto5 56878"
evals="minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,popqa::hamish_zs_reasoning,mmlu:cot::hamish_zs_reasoning,alpaca_eval_v3::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,mbppplus:0-shot-chat::tulu-thinker,codex_humanevalplus:0-shot-chat-v1::tulu-thinker"
python mason.py \
    --cluster ai2/saturn-cirrascale \
    --workspace ai2/tulu-thinker \
    --priority high \
    --image ai2/cuda12.8-dev-ubuntu22.04-torch2.6.0 \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --budget ai2/oe-adapt \
    --gpus 8 -- \
source configs/beaker_configs/ray_node_setup.sh \&\& \
source configs/beaker_configs/code_api_setup.sh \&\& \
python open_instruct/grpo_fast.py \
    --add_bos True \
    --exp_name ${exp_name} \
    --beta 0.0 \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 256 \
    --num_mini_batches 4 \
    --num_epochs 1 \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --output_dir /output \
    --kl_estimator kl3 \
    --dataset_mixer_list ${dataset_mix} \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/tulu_3_rewritten_100k 32 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 10240 \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 20480 \
    --model_name_or_path ${model_name_or_path} \
    --chat_template_name tulu_thinker_r1_style \
    --stop_strings "</answer>" \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 131072 \
    --deepspeed_stage 2 \
    --num_learners_per_node 6 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 2 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --num_evals 1 \
    --save_freq 32 \
    --try_launch_beaker_eval_jobs_on_weka True \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --oe_eval_max_length 32768 \
    --clip_higher 0.28 \
    --mask_truncated_completions True \
    --oe_eval_tasks ${evals}
