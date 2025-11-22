# beaker experiment create configs/judge_configs/proc_judge.yaml \
#     --name vllm_judge_qwen3-8b_v1_binary_LC \
#     --workspace ai2/oe-data

JUDGE_BASE_URL=http://saturn-cs-aus-476.reviz.ai2.in:8001/v1

python mason.py \
    --cluster ai2/jupiter \
    --task_name grpo_qwen3-8b-inst_v1_binary_LC_sym_r3 \
    --description "GRPO Qwen3-8B, v1 binary (length control symmetric) r3" \
    --workspace ai2/oe-data \
    --priority high \
    --image nathanl/open_instruct_auto \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-base \
    --env HOSTED_VLLM_API_BASE=${JUDGE_BASE_URL} \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --secret HF_TOKEN=yapeic_HF_TOKEN \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --exp_name grpo_qwen3-8b-inst_v1_binary_LC_sym_r3 \
    --beta 0.0 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 8 \
    --kl_estimator kl3 \
    --learning_rate 5e-7 \
    --dataset_mixer_list /weka/oe-training-default/yapeic/proc-data/data/dclm/tutorial_subset/batch_runs_prefiltered_4_15_new_filtered_v14_pp_v11_tools_v5_ff_v2/grpo_data/v1_binary-format_100k_LC.jsonl 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list /weka/oe-training-default/yapeic/proc-data/data/dclm/tutorial_subset/batch_runs_prefiltered_4_15_new_filtered_v14_pp_v11_tools_v5_ff_v2/grpo_data/v1_binary-format_100k_test_LC.jsonl 1.0 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 1024 \
    --response_length 2048 \
    --pack_length 3072 \
    --model_name_or_path Qwen/Qwen3-8B \
    --apply_verifiable_reward True \
    --llm_judge_model "hosted_vllm/yapeichang/distill_judge_qwen3-8b_sft_v6" \
    --non_stop_penalty True \
    --non_stop_penalty_value 0.0 \
    --temperature 1.0 \
    --total_episodes 200000 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 1 \
    --num_mini_batches 1 \
    --num_learners_per_node 4 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 4 \
    --lr_scheduler_type linear \
    --seed 1 \
    --local_eval_every 110 \
    --checkpoint_state_freq 40 \
    --checkpoint_state_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen3-8b-inst_v1_binary_LC_sym_r3_checkpoint_states \
    --output_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen3-8b-inst_v1_binary_LC_sym_r3_checkpoints \
    --save_freq 100 \
    --keep_last_n_checkpoints 10 \
    --gradient_checkpointing \
    --vllm_gpu_memory_utilization 0.7 \
    --with_tracking \
    --hf_entity yapeichang \
    --hf_repo_id grpo_qwen3-8b-inst_v1_binary_LC_sym_r3 \
    --wandb_project_name yapeic-exp \
    --wandb_entity ai2-llm