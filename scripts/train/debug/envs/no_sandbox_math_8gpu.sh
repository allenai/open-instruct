#!/bin/bash
# Baseline: GRPO math training WITHOUT sandbox environment (8 GPUs on Beaker)
#
# Same model, dataset, and hyperparams as sandbox_lm_8gpu.sh but without
# the sandbox environment — the model generates answers directly using
# chain-of-thought reasoning instead of executing code.
#
# This serves as a baseline to measure the effect of sandbox tool use.

# Get the Beaker username to construct the image name
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "No-sandbox GRPO math baseline with Qwen3-4B-Instruct" \
       --pure_docker_mode \
       --workspace ai2/olmo-instruct \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --budget ai2/oe-adapt \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list allenai/Dolci-RLZero-Math-7B 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 30720 \
    --pack_length 32768 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 8 \
    --inflight_updates true \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --temperature 1.0 \
    --learning_rate 3e-7 \
    --total_episodes 128000 \
    --deepspeed_stage 3 \
    --sequence_parallel_size 4 \
    --num_epochs 1 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 42 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --push_to_hub false \
    --with_tracking \
    --save_traces \
    --apply_verifiable_reward true \
    --stop_strings "</answer>" \
    --chat_template_name r1_simple_chat_postpend_think \
    --active_sampling \
    --rollouts_save_path /output/rollouts \
    --output_dir /output \
    --exp_name no_sandbox_math_qwen3_4b_grpo \
    --local_eval_every 10 \
    --save_freq 100 \
    --try_launch_beaker_eval_jobs_on_weka False
