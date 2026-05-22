#!/bin/bash

# Local 4-GPU debug run for dr-tulu RL (GRPO).
# Layout: 2 learner GPUs + 2 vLLM engine GPUs. Closer to production config.
# Uses Qwen3-0.6B + small dataset slice. No Ray head setup needed for single-node.
#
# Requires API keys in the environment:
#   export OPENAI_API_KEY=...
#   export SERPER_API_KEY=...
#   export JINA_API_KEY=...
#   export S2_API_KEY=...

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export RUBRIC_JUDGE_MODEL="${RUBRIC_JUDGE_MODEL:-gpt-4.1}"
export RUBRIC_GENERATION_MODEL="${RUBRIC_GENERATION_MODEL:-gpt-4.1}"

uv run open_instruct/grpo_fast.py \
    --exp_name rl_drtulu_local_4gpu \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_mixer_list rl-research/dr-tulu-rl-data 32 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list rl-research/dr-tulu-rl-data 8 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 1024 \
    --response_length 4096 \
    --pack_length 5120 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --learning_rate 5e-7 \
    --lr_scheduler_type constant \
    --total_episodes 64 \
    --deepspeed_stage 3 \
    --num_learners_per_node 2 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_enforce_eager \
    --beta 0.001 \
    --kl_estimator 3 \
    --load_ref_policy True \
    --async_steps 2 \
    --active_sampling \
    --inflight_updates \
    --temperature 1.0 \
    --non_stop_penalty False \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --apply_verifiable_reward true \
    --apply_evolving_rubric_reward true \
    --max_active_rubrics 5 \
    --remap_verifier general_rubric=rubric \
    --tool_parser_type vllm_qwen3_xml \
    --tools serper_search jina_browse s2_search \
    --tool_call_names google_search browse_webpage snippet_search \
    --tool_configs '{}' '{}' '{}' \
    --pool_size 32 \
    --backend_timeout 300 \
    --system_prompt_override_file scripts/train/dr-tulu/dr_tulu_adjusted.txt \
    --gradient_checkpointing \
    --local_eval_every 8 \
    --save_traces \
    --logging_steps 1 \
    --seed 1 \
    --report_to none \
    --with_tracking \
    --push_to_hub false
