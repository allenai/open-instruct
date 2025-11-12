# beaker experiment create configs/judge_configs/proc_judge.yaml \
#     --name vllm_judge_2 \
#     --workspace ai2/oe-data

JUDGE_BASE_URL=http://saturn-cs-aus-232.reviz.ai2.in:8001/v1

python mason.py \
    --cluster ai2/jupiter \
    --task_name grpo_qwen25-7b-inst_v3_ratio_10k_with_stop_fixed_template \
    --description "GRPO Qwen2.5-7B, ratio 10k, with stop, fixed template" \
    --workspace ai2/oe-data \
    --priority high \
    --image nathanl/open_instruct_auto \
    --preemptible \
    --num_nodes 2 \
    --budget ai2/oe-base \
    --env HOSTED_VLLM_API_BASE=${JUDGE_BASE_URL} \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --secret HF_TOKEN=yapeic_HF_TOKEN \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --exp_name grpo_qwen25-7b-inst_v3_ratio_10k_with_stop_fixed_template \
    --beta 0.0 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 16 \
    --kl_estimator kl3 \
    --learning_rate 5e-7 \
    --dataset_mixer_list /weka/oe-training-default/yapeic/proc-data/data/dclm/tutorial_subset/grpo_data/v3_goal_resource_steps_low_variation_simple_template_ratio_with_format_10k.jsonl 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list /weka/oe-training-default/yapeic/proc-data/data/dclm/tutorial_subset/grpo_data/v3_goal_resource_steps_low_variation_simple_template_ratio_with_format_10k_test.jsonl 1.0 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 1024 \
    --max_prompt_token_length 1024 \
    --response_length 4096 \
    --pack_length 5120 \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --stop_strings "</answer>" \
    --chat_template_name olmo_thinker_r1_style \
    --apply_verifiable_reward True \
    --llm_judge_model "hosted_vllm/yapeichang/distill_judge_qwen3-8b_sft_v2_fixed_data" \
    --non_stop_penalty True \
    --non_stop_penalty_value 0.0 \
    --temperature 1.0 \
    --total_episodes 160000 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 1 \
    --num_mini_batches 1 \
    --num_learners_per_node 8 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 8 \
    --lr_scheduler_type linear \
    --seed 1 \
    --local_eval_every 40 \
    --checkpoint_state_freq 20 \
    --checkpoint_state_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen25-7b-inst_v3_ratio_10k_with_stop_fixed_template_checkpoint_states \
    --save_freq 200 \
    --keep_last_n_checkpoints 10 \
    --gradient_checkpointing \
    --with_tracking \
    --hf_entity yapeichang \
    --hf_repo_id grpo_qwen25-7b-inst_v3_ratio_10k_with_stop_fixed_template \
    --wandb_project_name yapeic-exp \
    --wandb_entity ai2-llm