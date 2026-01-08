#!/bin/bash
# Tool-enabled inference on allenai/simpleqa_full
# Uses the same tool setup as scripts/train/debug/tool_grpo_fast.sh
model_name_or_path=$1
export SERPER_API_KEY=$(beaker secret read hamishivi_SERPER_API_KEY --workspace ai2/olmo-instruct)

# Build the command
uv run python -m open_instruct.tools.tool_inference \
    --model_name_or_path $model_name_or_path \
    --dataset allenai/simpleqa_full \
    --split test \
    --messages_key messages \
    --output_file outputs/simpleqa_generations.jsonl \
    --temperature 0.7 \
    --max_new_tokens 16384 \
    --tensor_parallel_size 1 \
    --tools serper_search python \
    --tool_configs '{}' '{\"api_endpoint\": \"https://open-instruct-tool-server-10554368204.us-central1.run.app/execute\"}' \
    --tool_tag_names search code \
    --tool_parser vllm_hermes \
    --max_tool_calls 5

