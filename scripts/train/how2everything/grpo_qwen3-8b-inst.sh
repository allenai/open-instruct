# GRPO training for Qwen3-8B with binary LLM judge reward (length-controlled)
#
# Prerequisites:
#   - 8 GPUs (1 node)
#   - A running vLLM judge server serving the judge model
#     (see configs/judge_configs/ for example deployment configs)
#   - Set HOSTED_VLLM_API_BASE to the judge server URL
#
# Usage:
#   export HOSTED_VLLM_API_BASE=http://<your-judge-server>:8001/v1
#   bash scripts/train/how2everything/grpo_qwen3-8b-inst.sh

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

source configs/beaker_configs/ray_node_setup.sh

python open_instruct/grpo_fast.py \
    --exp_name grpo_qwen3-8b-inst \
    --beta 0.0 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 8 \
    --kl_estimator kl3 \
    --learning_rate 5e-7 \
    --dataset_mixer_list how2everything/how2train_rl_100k 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list how2everything/how2train_rl_100k 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 1024 \
    --response_length 2048 \
    --pack_length 3072 \
    --model_name_or_path Qwen/Qwen3-8B \
    --apply_verifiable_reward True \
    --llm_judge_model "hosted_vllm/how2everything/how2judge" \
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
    --checkpoint_state_dir output/grpo_qwen3-8b-inst_checkpoint_states \
    --output_dir output/grpo_qwen3-8b-inst_checkpoints \
    --save_freq 100 \
    --keep_last_n_checkpoints 10 \
    --gradient_checkpointing \
    --vllm_gpu_memory_utilization 0.7 \
    --with_tracking
