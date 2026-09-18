#!/bin/bash

# Beaker OPD real-dump audit: runs the cheap Qwen3-0.6B <- Qwen3-1.7B pure-OPD
# smoke (scripts/train/debug/grpo_fast_opd_beaker.sh) with `--save_traces` so every
# training step dumps the rollout logprobs, the combined teacher logprobs, the
# response mask and the advantages the trainer consumed, then re-derives the
# advantage offline with `open_instruct.opd_trace_audit` and fails the job if the
# identity `advantage == kl_coef * (teacher - rollout)` or the `[:, 1:]` alignment
# does not hold on real data. Mirrors `open_instruct.miles.opd_audit`.
#
# 1 node x 2 GPUs (1 learner + 1 engine), 16 training steps, ~20 minutes.
# Launch with ./scripts/train/build_image_and_launch.sh so the image carries the
# current commit; the trace directory is printed at launch and in the job log.
# When the image cannot be rebuilt locally, set CODE_REF=<pushed branch or commit>:
# the job then clones that ref of `git remote get-url origin` and copies its
# `open_instruct/` package over the image's editable install (`/stage/open_instruct`,
# the same overlay `scripts/general_agent/terminal/rl/qwen35_math_posthoc_eval.sh`
# uses), so the code under test matches the ref. A PYTHONPATH overlay is not enough:
# the image's editable install resolves `open_instruct.*` imports to `/stage` first.

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
TRACE_DIR="${TRACE_DIR:-/weka/oe-adapt-default/allennlp/deletable_rollouts/opd_trace_audit/$(date -u +%Y%m%dT%H%M%SZ)}"
CODE_REF="${CODE_REF:-}"
SETUP="source configs/beaker_configs/ray_node_setup.sh"
if [[ -n "$CODE_REF" ]]; then
    ORIGIN_URL=$(git remote get-url origin | sed -E 's#^git@github.com:#https://github.com/#')
    # Plain && here: an unquoted $SETUP expansion yields literal "&&" words for mason to join.
    SETUP="git clone --depth 1 --branch $CODE_REF $ORIGIN_URL /tmp/oi && cp -r /tmp/oi/open_instruct/. /stage/open_instruct/ && $SETUP"
fi

echo "Using Beaker image: $BEAKER_IMAGE"
echo "Trace directory: $TRACE_DIR"
[[ -n "$CODE_REF" ]] && echo "Running code from $CODE_REF"

uv run python mason.py \
       --cluster ai2/jupiter \
       --cluster ai2/saturn \
       --image "$BEAKER_IMAGE" \
       --description "OPD real-dump audit: Qwen3-0.6B student + Qwen3-1.7B teacher, gsm8k, traces -> opd_trace_audit" \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --budget ai2/oe-other \
       --gpus 2 \
       --no_auto_dataset_cache \
       -- $SETUP \&\& python open_instruct/grpo_fast.py \
    --exp_name opd_trace_audit_qwen3 \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --opd_teacher_model_name_or_path Qwen/Qwen3-1.7B \
    --opd_kl_coef 1.0 \
    --opd_pure \
    --filter_zero_std_samples false \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 512 \
    --deepspeed_stage 3 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --load_ref_policy false \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --loss_fn dppo \
    --dppo_divergence_type tv \
    --dppo_divergence_threshold 0.1 \
    --use_liger_grpo_loss \
    --liger_grpo_loss_chunk_size 8 \
    --advantage_normalization_type centered \
    --seed 3 \
    --local_eval_every -1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --save_traces true \
    --rollouts_save_path "$TRACE_DIR" \
    --push_to_hub false \
    \&\& python -m open_instruct.opd_trace_audit --trace_dir "$TRACE_DIR" \
    --step 1 --step 2 --step 8 --step 16 --kl_coef 1.0 --output "$TRACE_DIR/audit.json"
