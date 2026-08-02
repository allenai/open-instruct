#!/bin/bash
# The cheapest thing that exercises the whole path: tiny policy, stubbed judge,
# no student endpoint, one GPU, twenty steps.
#
# It will not teach anything. It tells you the plugin loaded, the rows parse,
# the group scorer runs inside the Ray actor, and the reward reaches the
# trainer - which is every integration point, and all of them fail loudly here
# instead of quietly at step 40 of a real run.
#
#   ./projects/tutor/scripts/smoke.sh

set -euo pipefail

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1

DATA=${DATA:-data/tutor_smoke.jsonl}

if [ ! -f "$DATA" ]; then
    echo "build $DATA first:"
    echo "  python -m projects.tutor.build_dataset --items <screened.jsonl> --out $DATA --env ''"
    exit 1
fi

# --env '' above: single-turn, so no student endpoint is needed. stub=true on
# the scorer: no judge endpoint either. The only thing running is the policy.
uv run python open_instruct/grpo_fast.py \
    --dataset_mixer_list "$DATA" 1.0 \
    --dataset_mixer_list_splits train \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --reward_plugins projects.tutor.plugin \
    --group_scorer 'tutor:stub=true' \
    --apply_verifiable_reward false \
    --max_prompt_token_length 768 \
    --response_length 512 \
    --pack_length 1536 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 2 \
    --num_samples_per_prompt_rollout 4 \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 80 \
    --max_steps 20 \
    --deepspeed_stage 2 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enforce_eager \
    --vllm_sync_backend gloo \
    --gradient_checkpointing \
    --single_gpu_mode \
    --seed 0 \
    --push_to_hub false \
    --save_traces \
    --dataset_skip_cache \
    --output_dir output/tutor_smoke

echo "smoke passed"
