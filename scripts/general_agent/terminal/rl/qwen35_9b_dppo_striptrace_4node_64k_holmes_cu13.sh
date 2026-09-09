#!/bin/bash

# "STRIP thinking traces" chat-template run -- 4-node / 32 GPU DPPO @ 64k, cu13/B300/holmes.
#
# *** POST-HOC CORRECTION (2026-08-20): this is NOT a training-side ablation. ***
# The GRPO/DPPO rollout loop applies the chat template exactly ONCE per episode, at dataset
# preprocessing, on a system+user-only conversation (rlvr_tokenize_v1; see
# docs/algorithms/rollout_loop_internals.md). Both templates render that identically -- their only
# diff is in the assistant-history branch, which never executes. Mid-rollout, model tokens
# (including <think> spans) are appended VERBATIM to the token-level current_prompt
# (vllm_utils.py), and tool outputs are injected via hardcoded role strings in
# VLLM_PARSERS["vllm_qwen3_xml"] (environments/tools/parsers.py) -- jinja-independent.
# => Training trajectories are distributionally IDENTICAL to the prod run. With the shared
#    seed 42, this run is a SEED-REPLICA of qwen35_9b_dppo_repro_4node_64k_holmes_cu13.sh,
#    useful as a run-to-run variance control.
#
# Where the template DOES matter: EVAL. Checkpoints inherit the tokenizer/template of
# MODEL/TOKENIZER below (stock Qwen/Qwen3.5-9B), whose jinja strips prior-turn reasoning when an
# agent (e.g. Vanillux2Agent via vLLM serve) re-renders messages each turn:
#   stock Qwen  : `{%- if loop.index0 > ns.last_query_index %}` -> prior-turn <think> STRIPPED
#   hamishivi   : `{%- if reasoning_content %}`                 -> <think> KEPT every turn
# (The repos are otherwise identical: same safetensors LFS OIDs, same configs; ONLY
# chat_template.jinja differs. tokenizer_config.json's template field is a stale copy -- the
# .jinja file overrides it.)
# Measured effect at step 120 (k=1, TB2.1): strip 0.1685 vs keep 0.2697 (paired -0.101+/-0.039,
# McNemar p=0.022); TBlite unaffected (0.540 vs 0.524). Eval-time stripping hurts long-horizon
# tasks; models trained with reasoning-in-context must be SERVED with the keep-traces template.
#
# Otherwise a byte-for-byte copy of qwen35_9b_dppo_repro_4node_64k_holmes_cu13.sh except
# MODEL/TOKENIZER, EXP_NAME, MIRROR_URL (refreshed) and a fresh checkpoint_state_dir.
# Launch by REUSING the prod image (no rebuild -- only CLI args changed):
#   bash scripts/general_agent/terminal/rl/qwen35_9b_dppo_striptrace_4node_64k_holmes_cu13.sh \
#        shashankg/open-instruct-integration-test-omni_agent_cuda13-cuda13

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

MODEL=Qwen/Qwen3.5-9B
TOKENIZER=Qwen/Qwen3.5-9B

EXP_NAME=swerl_qwen35_9b_dppo_striptrace_4node_64k_holmes

uv run --no-default-groups --group dev --group cuda13 python mason.py \
       --cluster ai2/holmes \
       --image "$BEAKER_IMAGE" \
       --description "tmax-15k DPPO Qwen35 9b CHAT-TEMPLATE ABLATION: stock Qwen template STRIPS prev-turn reasoning (control vs hamishivi keep-traces prod run); 4-node; 64k; holmes/cu13/B300" \
       --pure_docker_mode \
       --workspace ai2/oe-agents-holmes \
       --priority urgent \
       --preemptible \
       --min_runtime 28800s \
       --auto_resume \
       --num_nodes 4 \
       --max_retries 5 \
       --env REPO_PATH=/stage \
       --env BEAKER_ALLOW_SUBCONTAINERS=1 \
       --env PYTORCH_ALLOC_CONF=expandable_segments:True \
       --env BEAKER_SKIP_DOCKER_SOCKET=1 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env DOCKERHUB_USERNAME=shashankg209 \
       --env SWERL_SANDBOX_TIMING_LOGS=1 \
       --env SWERL_RESET_FAILURE_ZERO_REWARD=1 \
       --env SWERL_DOCKER_AUTO_REMOVE=1 \
       --env SWERL_PODMAN_SERVICE_COUNT=4 \
       --env SWERL_DOCKER_START_CONCURRENCY=64 \
       --env SWERL_DOCKER_EXEC_CONCURRENCY=256 \
       --env SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S=1.0 \
       --env SWERL_PODMAN_IMAGE_JANITOR_ENABLED=1 \
       --env SWERL_PODMAN_IMAGE_JANITOR_INTERVAL_S=60 \
       --env SWERL_PODMAN_IMAGE_JANITOR_UNTIL=10m \
       --env MIRROR_URL=jupiter-cs-aus-208.reviz.ai2.in:5000 \
       --env PODMAN_NUM_LOCKS=65536 \
       --env CONTAINERS_STORAGE_CONF=/etc/containers/storage.conf \
       --secret DOCKER_PAT=shashankg_DOCKER_PAT \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- source scripts/docker/docker_login.sh \&\& source configs/beaker_configs/ray_node_setup.sh  \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list allenai/tmax-15k-open-instruct 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --per_turn_max_tokens 16384 \
    --response_length 65536 \
    --pack_length 67584 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 32 \
    --async_steps 4 \
    --model_name_or_path $MODEL \
    --tokenizer_name_or_path $TOKENIZER \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 128000 \
    --lr_scheduler_type constant \
    --deepspeed_stage 3 \
    --sequence_parallel_size 4 \
    --attn_implementation flash_4 \
    --num_epochs 1 \
    --num_learners_per_node 8 8 \
    --vllm_num_engines 16 \
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
    --save_trainer_logprobs true \
    --tools swerl_vanillux_sandbox \
    --tool_configs '{"task_data_hf_repo": "allenai/tmax-15k-open-instruct", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 512 \
    --max_steps 64 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
    --active_sampling \
    --backend_timeout 1200 \
    --vllm_gdn_prefill_backend triton \
    --checkpoint_state_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/shashankg/striptrace_4n64k_001 \
    --checkpoint_state_freq 5 \
    --inflight_updates true \
    --lm_head_fp32 true \
    --use_liger_grpo_loss \
    --liger_grpo_loss_chunk_size 8 \
    --advantage_normalization_type centered \
    --loss_fn dppo \
    --dppo_divergence_type tv \
    --dppo_divergence_threshold 0.1 \
    --rollouts_save_path /weka/oe-adapt-default/allennlp/deletable_rollouts/ \
    --output_dir /output \
    --exp_name $EXP_NAME \
    --local_eval_every 10 \
    --save_freq 20 \
    --try_launch_beaker_eval_jobs_on_weka False
