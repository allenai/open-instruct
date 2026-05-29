#!/bin/bash
# GFPO baseline runner for Qwen3-4B-Base (RL-Zero).
#
# Group Filtered Policy Optimization (Shrivastava et al., 2025, arXiv:2508.09726):
# oversample G responses per prompt, train only on the top-k ranked by a filter
# metric (shortest length, or token efficiency = reward/length). See
# open_instruct/gfpo.py. This is the published comparison method for our
# length-aware reward shaping; it leaves the reward untouched and intervenes
# only at advantage estimation.
#
# Mirrors cse-579-scripts/length_shaping_rl_qwen.sh exactly except:
#   - num_samples_per_prompt_rollout = G (default 16) instead of 8 (oversampling).
#   - total_episodes scaled so the #optimizer-steps still equals the baseline's
#     1000 (= NUM_UNIQUE_PROMPTS * G * STEPS), for a fair step-count comparison.
#     The paper likewise fixes step count across group sizes.
#   - No --length_reward_* args (GFPO and shaping are mutually exclusive).
#   - --gfpo_filter_metric / --gfpo_retain_k instead.
#   - --inflight_updates True: at G=16 the per-step generation (1024 long
#     sequences) takes >120s to drain, so the default (drain-then-sync) weight
#     sync hits the hardcoded WEIGHT_SYNC_TIMEOUT_S and the run dies at step ~2.
#     inflight_updates lets vLLM pause mid-generation for the sync.
#
# Required env vars:
#   GFPO_METRIC    shortest | token_efficiency
# Optional env vars:
#   GROUP_SIZE     G, responses sampled per prompt        (default: 16)
#   RETAIN_K       k, responses retained for gradient      (default: 8)
#   STEPS          optimizer steps to match the baseline   (default: 1000)
#   GIT_REF        branch in allenai/open-instruct         (default: jacobm/cse-579)
#   EXP_SUFFIX     appended to exp_name                    (default: empty)
#   SECRETS_WORKSPACE  workspace to read secrets from      (default: ai2/ianm)
#
# Example (shortest-8/16):
#   GFPO_METRIC=shortest bash cse-579-scripts/gfpo_rl_qwen.sh

set -euo pipefail

: "${GFPO_METRIC:?GFPO_METRIC is required (shortest|token_efficiency)}"
GROUP_SIZE=${GROUP_SIZE:-16}
RETAIN_K=${RETAIN_K:-8}
STEPS=${STEPS:-1000}
GIT_REF=${GIT_REF:-jacobm/cse-579}
EXP_SUFFIX=${EXP_SUFFIX:-}

# Keep optimizer-step count equal to the baseline (1000 steps * 64 prompts).
NUM_UNIQUE_PROMPTS=64
TOTAL_EPISODES=$((NUM_UNIQUE_PROMPTS * GROUP_SIZE * STEPS))

SECRETS_WORKSPACE=${SECRETS_WORKSPACE:-ai2/ianm}
LAUNCH_WANDB_KEY=$(beaker secret read -w "$SECRETS_WORKSPACE" IANM_WANDB_API_KEY)
LAUNCH_HF_TOKEN=$(beaker secret read -w "$SECRETS_WORKSPACE" HF_TOKEN)
LAUNCH_BEAKER_TOKEN=$(beaker secret read -w "$SECRETS_WORKSPACE" IANM_BEAKER_TOKEN)

EXP_NAME="gfpo_qwen_4b_base_mixed_${GFPO_METRIC}_g${GROUP_SIZE}k${RETAIN_K}${EXP_SUFFIX}"

CHECKPOINT_STATE_DIR=${CHECKPOINT_STATE_DIR:-/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/ianm/${EXP_NAME}_$(date +%s)}

uv run python mason.py \
    --budget ai2/oe-other \
    --cluster ai2/jupiter \
    --image nathanl/open_instruct_auto \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --num_nodes 1 \
    --gpus 8 \
    --max_retries 0 \
    --no_auto_dataset_cache \
    --env RAY_CGRAPH_get_timeout=300 \
    --env "LAUNCH_WANDB_KEY=$LAUNCH_WANDB_KEY" \
    --env "LAUNCH_HF_TOKEN=$LAUNCH_HF_TOKEN" \
    --env "LAUNCH_BEAKER_TOKEN=$LAUNCH_BEAKER_TOKEN" \
    -- export WANDB_API_KEY=\$LAUNCH_WANDB_KEY HF_TOKEN=\$LAUNCH_HF_TOKEN BEAKER_TOKEN=\$LAUNCH_BEAKER_TOKEN \&\& rm -rf /stage/open_instruct \&\& git clone --depth=1 -b "$GIT_REF" https://github.com/allenai/open-instruct.git /tmp/oi_branch \&\& cp -r /tmp/oi_branch/open_instruct /stage/ \&\& source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
        --exp_name "$EXP_NAME" \
        --beta 0.0 \
        --num_samples_per_prompt_rollout "$GROUP_SIZE" \
        --num_unique_prompts_rollout "$NUM_UNIQUE_PROMPTS" \
        --num_mini_batches 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --output_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint/ianm/ \
        --kl_estimator 2 \
        --dataset_mixer_list jacobmorrison/cse-579-mixed-rl 1.0 \
        --dataset_mixer_list_splits train \
        --max_prompt_token_length 2048 \
        --response_length 30720 \
        --pack_length 32768 \
        --model_name_or_path Qwen/Qwen3-4B-Base \
        --chat_template_name qwen_instruct_user_boxed_math \
        --non_stop_penalty False \
        --mask_truncated_completions False \
        --temperature 1.0 \
        --total_episodes "$TOTAL_EPISODES" \
        --inflight_updates True \
        --deepspeed_stage 3 \
        --num_learners_per_node 4 \
        --sequence_parallel_size 1 \
        --vllm_num_engines 4 \
        --vllm_tensor_parallel_size 1 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --local_eval_every 100 \
        --save_freq 100 \
        --eval_priority urgent \
        --eval_workspace ai2/olmo-instruct \
        --try_launch_beaker_eval_jobs_on_weka True \
        --gradient_checkpointing \
        --with_tracking \
        --code_api_url https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program \
        --code_pass_rate_reward_threshold 0.99 \
        --oe_eval_max_length 32768 \
        --checkpoint_state_freq 100 \
        --checkpoint_state_dir "$CHECKPOINT_STATE_DIR" \
        --oe_eval_gpu_multiplier 4 \
        --keep_last_n_checkpoints -1 \
        --gfpo_filter_metric "$GFPO_METRIC" \
        --gfpo_retain_k "$RETAIN_K" \
        --oe_eval_beaker_image oe-eval-beaker/oe_eval_olmo2_retrofit_auto \
        --oe_eval_tokenizer_path /weka/oe-adapt-default/jacobm/repos/cse-579/tokenizers/qwen3-olmo-thinker-eos-old-transformers \
        --oe_eval_tasks alpaca_eval_v3::hamish_zs_reasoning_deepseek,minerva_math_500::hamish_zs_reasoning,ifbench::tulu,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2025_deepseek
