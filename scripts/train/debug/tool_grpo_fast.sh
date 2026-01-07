#!/bin/bash

# Get the Beaker username to construct the image name
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/jupiter \
       --cluster ai2/ceres \
       --image "$BEAKER_IMAGE" \
       --description "Multi-node tool use training test (2 nodes, 8 GPUs each)." \
       --pure_docker_mode \
       --workspace ai2/tulu-thinker \
       --priority high \
       --preemptible \
       --num_nodes 2 \
       --max_retries 0 \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --budget ai2/oe-adapt \
       --secret SERPER_API_KEY=hamishivi_SERPER_API_KEY \
       --gpus 8 \
	   -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/rl_rag_shortformqa 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/rl_rag_shortformqa 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --active_sampling \
    --inflight_updates \
    --async_steps 8 \
    --pack_length 18432 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-8B \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --ground_truths_key ground_truth \
    --learning_rate 3e-7 \
    --total_episodes 32000 \
    --deepspeed_stage 3 \
    --with_tracking \
    --num_epochs 1 \
    --num_learners_per_node 8 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 8 \
    --beta 0.01 \
    --seed 3 \
    --local_eval_every 1 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --output_dir /output \
    --tools serper_search python \
    --tool_configs '{}' '{"api_endpoint": "https://open-instruct-tool-server-10554368204.us-central1.run.app/execute"}' \
    --tool_tag_names search code \
    --tool_parser vllm_hermes \
    --push_to_hub false
