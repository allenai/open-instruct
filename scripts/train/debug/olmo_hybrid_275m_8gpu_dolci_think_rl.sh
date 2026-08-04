#!/bin/bash

# 8 GPU GRPO (OLMo-core grpo.py) Dolci-Think-RL run for the OLMo-hybrid small suite
# (275M SFT checkpoint): 4 GPUs for training, 4 GPUs for generation.
# Requires a Beaker image built from the transformers / vLLM / OLMo-core
# maintainer forks that implement the `olmo_hybrid_small` architecture (see
# pyproject.toml [tool.uv.sources]).
# Points at the HF-converted checkpoint (-hf suffix); grpo.py deserializes the
# sibling olmo-core-native checkpoint's own TransformerConfig to build the model.
# Uses local evals (aime25, a lcbv6 subset, and an IF eval) every 200 steps
# instead of oe-eval.
#
# Batch-size defaults (num_samples_per_prompt_rollout, num_unique_prompts_rollout,
# learning_rate, max_prompt_token_length) taken from scripts/train/olmo3/7b_think_rl.sh.
# Pack length bumped to 32k (response_length 30720 + max_prompt_token_length 2048,
# matching scripts/train/qwen/qwen3_4b_dapo_math_32k.sh).
#
# Inspired by scripts/train/debug/olmo_hybrid_275m_4gpu_gsm8k.sh.

EXP_NAME="olmo_hybrid_275m_dolci_think_rl"
DATASETS="allenai/Dolci-Think-RL-7B 1.0"
EVAL_DATASETS="mnoukhov/aime_2025_openinstruct 1.0 mnoukhov/sdpo_lcbv6 1.0 allenai/IF_multi_constraints_upto5 32"
EVAL_SPLITS="train test train"
MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-sft-think-275M-lr2e-4/step23206-hf

# Default to this branch's cuda12 integration test image, as built by
# scripts/train/build_image_and_launch.sh --cuda-version 12.
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
SANITIZED_BRANCH=$(echo "$GIT_BRANCH" | sed 's/[^a-zA-Z0-9._-]/-/g' | tr '[:upper:]' '[:lower:]' | sed 's/^-//')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test-${SANITIZED_BRANCH}-cuda12}"
[[ $# -gt 0 ]] && shift

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --task_name ${EXP_NAME} \
    --cluster ai2/jupiter \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --pure_docker_mode \
    --image "$BEAKER_IMAGE" \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    --no_auto_dataset_cache \
    --artifact_ttl 1d \
    --gpus 8 \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo.py \
    --exp_name ${EXP_NAME} \
    --run_name ${EXP_NAME} \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $EVAL_DATASETS \
    --dataset_mixer_eval_list_splits $EVAL_SPLITS \
    --beta 0.0 \
    --async_steps 4 \
    --inflight_updates \
    --active_sampling \
    --filter_zero_std_samples True \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 8 \
    --num_unique_prompts_rollout 64 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --max_prompt_token_length 2048 \
    --response_length 30720 \
    --pack_length 32768 \
    --model_name_or_path "$MODEL_PATH" \
    --chat_template_name olmo_thinker \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 512000 \
    --fsdp_shard_degree 4 \
    --fsdp_num_replicas 1 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 200 \
    --save_freq 200 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --mask_truncated_completions False \
    --load_ref_policy True \
    --with_tracking \
    --push_to_hub False "$@"
