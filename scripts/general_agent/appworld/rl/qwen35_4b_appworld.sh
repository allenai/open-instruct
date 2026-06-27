#!/bin/bash

# AppWorld code-as-action RL with Qwen3.5-4B (GRPO + appworld env).
#
# Modeled on scripts/general_agent/terminal/rl/qwen35_4b_base_tmax_10k.sh (the swerl
# podman-services recipe) — AppWorld reuses the same per-rollout container infra, but
# each rollout runs an AppWorld server container and the env talks to it over HTTP.
#
# Reward is AppWorld's own /evaluate (fraction of unit tests passed), emitted by the env;
# the dataset uses the `passthrough` verifier so no extra reward is added. The prompt
# (incl. supervisor creds) is baked into the dataset messages, so NO system_prompt_override.
#
# Data: the AppWorld task data is baked into the container image (shatu/appworld-data:latest,
# data_root="" => no bind mount). The image is pulled through the cluster registry mirror.
#
# Launch (dirty tree is fine via the _dirty variant):
#   ./scripts/train/build_image_and_launch_dirty.sh scripts/general_agent/appworld/rl/qwen35_4b_appworld.sh

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"
MODEL=Qwen/Qwen3.5-4B
TOKENIZER=Qwen/Qwen3.5-4B
DATASET=shatu/appworld-rl
APPWORLD_IMAGE=shatu/appworld-data:latest
MAX_STEPS=30

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "AppWorld GRPO with Qwen3.5-4B" \
       --pure_docker_mode \
       --workspace ai2/general-tool-use \
       --priority urgent \
       --preemptible \
       --num_nodes 2 \
       --max_retries 5 \
       --env REPO_PATH=/stage \
       --env BEAKER_ALLOW_SUBCONTAINERS=1 \
       --env BEAKER_SKIP_DOCKER_SOCKET=1 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env DOCKERHUB_USERNAME=shashankg209 \
       --env SWERL_SANDBOX_TIMING_LOGS=1 \
       --env SWERL_DOCKER_AUTO_REMOVE=1 \
       --env SWERL_RESET_FAILURE_ZERO_REWARD=1 \
       --env SWERL_PODMAN_SERVICE_COUNT=8 \
       --env SWERL_DOCKER_START_CONCURRENCY=128 \
       --env SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S=1.0 \
       --env MIRROR_URL=jupiter-cs-aus-193.reviz.ai2.in:5000 \
       --env PODMAN_NUM_LOCKS=65536 \
       --env CONTAINERS_STORAGE_CONF=/etc/containers/storage.conf \
       --secret DOCKER_PAT=shashankg_DOCKER_PAT \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source scripts/docker/docker_login.sh \&\& source configs/beaker_configs/ray_node_setup.sh  \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list $DATASET 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $DATASET 8 \
    --dataset_mixer_eval_list_splits dev \
    --max_prompt_token_length 2048 \
    --per_turn_max_tokens 8192 \
    --response_length 24576 \
    --pack_length 26624 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 4 \
    --model_name_or_path $MODEL \
    --tokenizer_name_or_path $TOKENIZER \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 64000 \
    --lr_scheduler_type constant \
    --deepspeed_stage 3 \
    --sequence_parallel_size 2 \
    --num_epochs 1 \
    --num_learners_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --seed 42 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --push_to_hub false \
    --with_tracking \
    --wandb_project oe-general-agents \
    --save_traces \
    --tools appworld \
    --tool_call_names execute_python \
    --tool_configs "{\"image\": \"${APPWORLD_IMAGE}\", \"data_root\": \"\", \"max_interactions\": ${MAX_STEPS}, \"startup_timeout\": 600, \"dense_reward\": true}" \
    --pool_size 128 \
    --max_steps ${MAX_STEPS} \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --active_sampling \
    --backend_timeout 1200 \
    --checkpoint_state_freq 10 \
    --inflight_updates true \
    --advantage_normalization_type centered \
    --rollouts_save_path /weka/oe-adapt-default/allennlp/deletable_rollouts/ \
    --output_dir /output \
    --exp_name appworld_qwen35_4b_grpo \
    --local_eval_every 10 \
    --save_freq 20 \
    --try_launch_beaker_eval_jobs_on_weka False
