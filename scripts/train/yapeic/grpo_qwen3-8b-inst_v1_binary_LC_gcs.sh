# beaker experiment create configs/judge_configs/proc_judge.yaml \
#     --name vllm_judge_qwen3-8b_v1_binary_LC_gcs \
#     --workspace ai2/oe-data

JUDGE_BASE_URL=http://saturn-cs-aus-242.reviz.ai2.in:8001/v1

python mason.py \
    --cluster ai2/augusta \
    --task_name grpo_qwen3-8b-inst_v1_binary_LC_gcs \
    --description "GRPO Qwen3-8B, v1 binary (length control)" \
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
    --exp_name grpo_qwen3-8b-inst_v1_binary_LC_gcs \
    --beta 0.0 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 8 \
    --kl_estimator kl3 \
    --learning_rate 5e-7 \
    --dataset_mixer_list yapeichang/v1_binary-format_100k_LC 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list yapeichang/v1_binary-format_100k_test_LC 1.0 \
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
    --checkpoint_state_freq 30 \
    --save_freq 100 \
    --keep_last_n_checkpoints 10 \
    --gradient_checkpointing \
    --vllm_gpu_memory_utilization 0.5 \
    --with_tracking \
    --output_dir /output/grpo_qwen3-8b-inst_v1_binary_LC_gcs/checkpoints \
    --checkpoint_state_dir /output/grpo_qwen3-8b-inst_v1_binary_LC_gcs/checkpoint_states \
    --gs_snapshots_dir gs://ai2-llm/checkpoints/yapeic/grpo_qwen3-8b-inst_v1_binary_LC_gcs/checkpoints \
    --gs_checkpoint_state_dir gs://ai2-llm/checkpoints/yapeic/grpo_qwen3-8b-inst_v1_binary_LC_gcs/checkpoint_states \
    --hf_entity yapeichang \
    --hf_repo_id grpo_qwen3-8b-inst_v1_binary_LC_gcs \
    --wandb_project_name yapeic-exp \
    --wandb_entity ai2-llm