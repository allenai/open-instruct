#!/bin/bash
# Single-GPU smoke test for length-aware reward shaping (~8 min).
#
# Adapted from scripts/train/debug/single_gpu_on_beaker.sh; uses the
# small rlvr_gsm8k_zs dataset and overlays our feature branch onto the
# bundled image at job start, so no local Docker build is needed.
#
# Defaults exercise linear decay with alpha=1.0; override via env vars.

set -euo pipefail

SHAPING_METHOD=${SHAPING_METHOD:-linear}
DECAY_PARAM=${DECAY_PARAM:-1.0}
WARMUP_TYPE=${WARMUP_TYPE:-constant}
WARMUP_FRACTION=${WARMUP_FRACTION:-0.25}
SOLVE_RATE_THRESHOLD=${SOLVE_RATE_THRESHOLD:-0.3}
GIT_REF=${GIT_REF:-ian/length-shaping}

EXP_NAME="lenshape_smoke_${SHAPING_METHOD}_p${DECAY_PARAM}"

uv run python mason.py \
       --cluster ai2/jupiter \
       --cluster ai2/saturn \
       --image nathanl/open_instruct_auto \
       --description "Length shaping smoke test (single GPU)." \
       --pure_docker_mode \
       --no-host-networking \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --num_nodes 1 \
       --max_retries 0 \
       --timeout 15m \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --budget ai2/oe-other \
       --gpus 1 \
       --no_auto_dataset_cache \
       -- rm -rf /stage/open_instruct \&\& git clone --depth=1 -b "$GIT_REF" https://github.com/allenai/open-instruct.git /tmp/oi_branch \&\& cp -r /tmp/oi_branch/open_instruct /stage/ \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --exp_name "$EXP_NAME" \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --add_bos \
    --stop_strings "</answer>" \
    --apply_r1_style_format_reward \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --inflight_updates True \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --learning_rate 3e-7 \
    --total_episodes 200 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --load_ref_policy true \
    --seed 3 \
    --local_eval_every 1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --push_to_hub false \
    --single_gpu_mode \
    --length_reward_shaping_method "$SHAPING_METHOD" \
    --length_reward_decay_param "$DECAY_PARAM" \
    --length_reward_warmup_type "$WARMUP_TYPE" \
    --length_reward_warmup_fraction "$WARMUP_FRACTION" \
    --length_reward_solve_rate_threshold "$SOLVE_RATE_THRESHOLD"
