#!/bin/bash
# 1-gpu training script for dr-tulu style tool use training.

# Note: replace with your own key if outside of ai2.
export SERPER_API_KEY=$(beaker secret read hamishivi_SERPER_API_KEY --workspace ai2/dr-tulu-ablations)
export CRAWL4AI_API_KEY=$(beaker secret read hamishivi_CRAWL4AI_API_KEY --workspace ai2/dr-tulu-ablations)
export S2_API_KEY=$(beaker secret read hamishivi_S2_API_KEY --workspace ai2/dr-tulu-ablations)
export CRAWL4AI_API_URL="http://kennel.csail.mit.edu:11236"  # shannons crawl4ai server.

uv sync --extra dr-tulu

# launch mcp server
MCP_CACHE_DIR=".cache-$(hostname)" uv run --extra dr-tulu python -m dr_agent.mcp_backend.main --port 8000 --host 0.0.0.0 --path /mcp &

# Run training
VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/tulu_3_rewritten_100k 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/tulu_3_rewritten_100k 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 512 \
    --active_sampling \
    --async_steps 8 \
    --pack_length 3072 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think_tool_vllm \
    --learning_rate 3e-7 \
    --total_episodes 200 \
    --deepspeed_stage 3 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 3 \
    --local_eval_every 1 \
    --vllm_sync_backend gloo \
    --single_gpu_mode \
    --vllm_gpu_memory_utilization 0.3 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --system_prompt_override_file scripts/train/debug/dr_tulu_system_prompt.txt \
    --tools mcp \
    --mcp_tool_names 'snippet_search,google_search,browse_webpage' \
    --mcp_parser_name v20250824 \
    --mcp_host 0.0.0.0 \
    --mcp_port 8000 \
    --push_to_hub false
