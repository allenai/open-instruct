#!/bin/bash
# Length-aware reward shaping experiment runner for Olmo 3 7B Think.
#
# Mirrors cse-579-scripts/baseline_rl_7b.sh exactly except for:
#   - --budget ai2/oe-other (Jacob uses ai2/oe-adapt; Ian's account uses oe-other)
#   - The command is prefixed with a git-clone overlay so the running container
#     picks up the shaping changes from the ian/length-shaping branch (the bundled
#     /stage/open_instruct/ in nathanl/open_instruct_auto predates these edits).
#   - Five extra --length_reward_* CLI args (the entire feature surface).
#   - exp_name encodes the shaping config so multiple runs don't collide.
#
# Required env vars:
#   SHAPING_METHOD     none|linear|exponential|rank|binary_shortest|soft_threshold
#   DECAY_PARAM        alpha (linear) / lambda (exponential) / threshold (soft_threshold)
# Optional env vars:
#   WARMUP_TYPE        constant|linear|solve_rate            (default: constant)
#   WARMUP_FRACTION    fraction of training to ramp over     (default: 0.25)
#   SOLVE_RATE_THRESHOLD                                     (default: 0.3)
#   GIT_REF            branch in allenai/open-instruct       (default: ian/length-shaping)
#   EXP_SUFFIX         appended to exp_name                  (default: empty)
#
# Example (linear decay, alpha=1.0):
#   SHAPING_METHOD=linear DECAY_PARAM=1.0 \
#     bash cse-579-scripts/length_shaping_rl_7b.sh

set -euo pipefail

: "${SHAPING_METHOD:?SHAPING_METHOD is required}"
: "${DECAY_PARAM:?DECAY_PARAM is required}"
WARMUP_TYPE=${WARMUP_TYPE:-constant}
WARMUP_FRACTION=${WARMUP_FRACTION:-0.25}
SOLVE_RATE_THRESHOLD=${SOLVE_RATE_THRESHOLD:-0.3}
GIT_REF=${GIT_REF:-ian/length-shaping}
EXP_SUFFIX=${EXP_SUFFIX:-}

# Read secrets from ai2/ianm and pass them through under aliases.
# Works around mason's per-workspace secret lookup (ai2/olmo-instruct does
# not yet have ianm_* secrets) AND Beaker's guard against literal --env
# values for canonical secret names (WANDB_API_KEY etc). We re-export to
# the canonical names inside the container before python starts.
# The values get embedded literally in the Beaker job spec, so do not use
# this pattern with secrets more sensitive than class-project keys.
SECRETS_WORKSPACE=${SECRETS_WORKSPACE:-ai2/ianm}
LAUNCH_WANDB_KEY=$(beaker secret read -w "$SECRETS_WORKSPACE" IANM_WANDB_API_KEY)
LAUNCH_HF_TOKEN=$(beaker secret read -w "$SECRETS_WORKSPACE" HF_TOKEN)
LAUNCH_BEAKER_TOKEN=$(beaker secret read -w "$SECRETS_WORKSPACE" IANM_BEAKER_TOKEN)

EXP_NAME="lenshape_olmo_7b_sft_mixed_${SHAPING_METHOD}_p${DECAY_PARAM}_w${WARMUP_TYPE}${EXP_SUFFIX}"

uv run python mason.py \
    --budget ai2/oe-other \
    --cluster ai2/jupiter \
    --image nathanl/open_instruct_auto \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
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
        --num_samples_per_prompt_rollout 8 \
        --num_unique_prompts_rollout 64 \
        --num_mini_batches 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --output_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint/ianm/ \
        --kl_estimator 2 \
        --dataset_mixer_list jacobmorrison/cse-579-mixed-rl 1.0 \
        --dataset_mixer_list_splits train \
        --max_prompt_token_length 2048 \
        --response_length 32768 \
        --pack_length 35840 \
        --model_name_or_path allenai/Olmo-3-7B-Instruct-SFT \
        --chat_template_name olmo_think \
        --non_stop_penalty False \
        --mask_truncated_completions False \
        --temperature 1.0 \
        --total_episodes 512000 \
        --deepspeed_stage 3 \
        --num_learners_per_node 8 8 \
        --sequence_parallel_size 1 \
        --vllm_num_engines 16 \
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
        --oe_eval_gpu_multiplier 4 \
        --keep_last_n_checkpoints -1 \
        --length_reward_shaping_method "$SHAPING_METHOD" \
        --length_reward_decay_param "$DECAY_PARAM" \
        --length_reward_warmup_type "$WARMUP_TYPE" \
        --length_reward_warmup_fraction "$WARMUP_FRACTION" \
        --length_reward_solve_rate_threshold "$SOLVE_RATE_THRESHOLD" \
        --oe_eval_beaker_image oe-eval-beaker/oe_eval_olmo2_retrofit_auto \
        --oe_eval_tasks alpaca_eval_v3::hamish_zs_reasoning_deepseek,minerva_math_500::hamish_zs_reasoning,ifbench::tulu,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2025_deepseek
