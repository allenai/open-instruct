#!/bin/bash
# ECHO-loss variant of olmo_3_parser_multigpu.sh: multinode GRPO tool-use test
# with the ECHO world-modeling loss (arXiv:2605.24517) enabled, on the
# production-shaped loss path (liger tiled loss + DPPO + sequence parallel).
# 1 training node (8 GPUs) + 1 inference node (8 GPUs).
# Success criteria: job completes 5 steps and wandb shows a nonzero
# loss/echo_nll_per_token and echo/observation_token_count.

# Get the Beaker username to construct the image name
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

export SERPER_API_KEY=$(beaker secret read ${BEAKER_USER}_SERPER_API_KEY --workspace ai2/open-instruct-dev)
export JINA_API_KEY=$(beaker secret read ${BEAKER_USER}_JINA_API_KEY --workspace ai2/open-instruct-dev)

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "OLMo-3 multinode tool use test with ECHO loss (liger+DPPO)" \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --preemptible \
       --num_nodes 2 \
       --max_retries 0 \
       --timeout 1h \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --budget ai2/oe-adapt \
       --secret SERPER_API_KEY=${BEAKER_USER}_SERPER_API_KEY \
       --secret JINA_API_KEY=${BEAKER_USER}_JINA_API_KEY \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/tulu_3_rewritten_100k 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/tulu_3_rewritten_100k 32 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 16384 \
    --inflight_updates True \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 8 \
    --model_name_or_path allenai/Olmo-3-7B-Instruct-SFT \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --exp_name olmo3_7b_tool_use_echo_test \
    --learning_rate 5e-7 \
    --total_episodes $((5 * 32 * 8)) \
    --deepspeed_stage 3 \
    --sequence_parallel_size 2 \
    --with_tracking \
    --num_epochs 1 \
    --num_learners_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --use_liger_grpo_loss \
    --liger_grpo_loss_chunk_size 8 \
    --loss_fn dppo \
    --dppo_divergence_type tv \
    --dppo_divergence_threshold 0.1 \
    --echo_loss_alpha 0.1 \
    --seed 1 \
    --local_eval_every 10 \
    --gradient_checkpointing \
    --push_to_hub false \
    --output_dir /output \
    --kl_estimator 2 \
    --non_stop_penalty False \
    --num_mini_batches 1 \
    --lr_scheduler_type constant \
    --save_freq 100 \
    --try_launch_beaker_eval_jobs_on_weka False \
    --max_steps 5 \
    --vllm_enable_prefix_caching \
    --tools python serper_search jina_browse \
    --tool_call_names code search browse \
    --tool_configs '{"api_endpoint": "https://open-instruct-tool-server-10554368204.us-central1.run.app/execute", "timeout": 3}' '{}' '{}' \
    --tool_parser_type vllm_olmo3
