#!/bin/bash
# Length-aware reward shaping experiment runner.
#
# Starts from Olmo-3-7B-Think-SFT (skipping DPO, matching the baseline that the
# lab mate is running) and applies dynamic length-aware reward shaping during
# RLVR. Configure the experiment via env vars; each combination corresponds to
# one row in the proposal experiment table.
#
# Required env vars:
#   SHAPING_METHOD     none|linear|exponential|rank|binary_shortest|soft_threshold
#   DECAY_PARAM        alpha (linear) / lambda (exponential) / threshold (soft_threshold)
# Optional env vars:
#   WARMUP_TYPE        constant|linear|solve_rate            (default: constant)
#   WARMUP_FRACTION    fraction of training to ramp over     (default: 0.25)
#   SOLVE_RATE_THRESHOLD                                     (default: 0.3)
#   EXP_SUFFIX         appended to exp_name for disambiguation
#   MODEL_NAME         starting checkpoint                   (default: allenai/Olmo-3-7B-Think-SFT)
#
# Example:
#   SHAPING_METHOD=linear DECAY_PARAM=1.0 ./scripts/train/build_image_and_launch.sh \
#       scripts/train/olmo3/7b_think_rl_length_shaping.sh

set -euo pipefail

BEAKER_IMAGE=${1:-nathanl/open_instruct_auto}

: "${SHAPING_METHOD:?SHAPING_METHOD is required}"
: "${DECAY_PARAM:?DECAY_PARAM is required}"
WARMUP_TYPE=${WARMUP_TYPE:-constant}
WARMUP_FRACTION=${WARMUP_FRACTION:-0.25}
SOLVE_RATE_THRESHOLD=${SOLVE_RATE_THRESHOLD:-0.3}
EXP_SUFFIX=${EXP_SUFFIX:-}
MODEL_NAME=${MODEL_NAME:-allenai/Olmo-3-7B-Think-SFT}
# Branch in allenai/open-instruct whose open_instruct/ directory will overlay
# the version baked into the base image. Only Python source is overlaid; if you
# add new pyproject.toml deps, also add a `uv sync` step below.
GIT_REF=${GIT_REF:-ian/length-shaping}

EXP_NAME="7b_olmo3_thinker_sft_to_rl_lenshape_${SHAPING_METHOD}_p${DECAY_PARAM}_w${WARMUP_TYPE}${EXP_SUFFIX}"

python mason.py \
    --budget ai2/oe-other \
    --cluster ai2/jupiter \
    --image nathanl/open_instruct_auto \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --num_nodes 9 \
    --gpus 8 \
    --max_retries 0 \
    --no_auto_dataset_cache \
    --env RAY_CGRAPH_get_timeout=300 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env HOSTED_VLLM_API_BASE=http://ceres-cs-aus-447.reviz.ai2.in:8001/v1 \
    -- rm -rf /stage/open_instruct \&\& git clone --depth=1 -b "$GIT_REF" https://github.com/allenai/open-instruct.git /tmp/oi_branch \&\& cp -r /tmp/oi_branch/open_instruct /stage/ \&\& source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
        --exp_name "$EXP_NAME" \
        --beta 0.0 \
        --num_samples_per_prompt_rollout 8 \
        --num_unique_prompts_rollout 64 \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --output_dir /output \
        --kl_estimator 2 \
        --dataset_mixer_list allenai/Dolci-Think-RL-7B 1.0 \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list allenai/Dolci-Think-RL-7B 8 \
        --dataset_mixer_eval_list_splits train \
        --max_token_length 10240 \
        --max_prompt_token_length 2048 \
        --response_length 32768 \
        --pack_length 35840 \
        --model_name_or_path "$MODEL_NAME" \
        --chat_template_name olmo_thinker \
        --non_stop_penalty False \
        --mask_truncated_completions False \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes 10000000 \
        --deepspeed_stage 3 \
        --num_learners_per_node 8 8 \
        --vllm_num_engines 56 \
        --vllm_tensor_parallel_size 1 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --local_eval_every 50 \
        --save_freq 25 \
        --beaker_eval_freq 50 \
        --eval_priority urgent \
        --try_launch_beaker_eval_jobs_on_weka True \
        --gradient_checkpointing \
        --with_tracking \
        --llm_judge_model hosted_vllm/Qwen/Qwen3-32B \
        --llm_judge_timeout 600 \
        --llm_judge_max_tokens 2048 \
        --llm_judge_max_context_length 32768 \
        --code_api_url https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program \
        --code_pass_rate_reward_threshold 0.99 \
        --oe_eval_max_length 32768 \
        --checkpoint_state_freq 100 \
        --backend_timeout 1200 \
        --inflight_updates false \
        --async_steps 1 \
        --length_reward_shaping_method "$SHAPING_METHOD" \
        --length_reward_decay_param "$DECAY_PARAM" \
        --length_reward_warmup_type "$WARMUP_TYPE" \
        --length_reward_warmup_fraction "$WARMUP_FRACTION" \
        --length_reward_solve_rate_threshold "$SOLVE_RATE_THRESHOLD" \
        --oe_eval_beaker_image oe-eval-beaker/oe_eval_olmo2_retrofit_auto \
        --oe_eval_tasks mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,simpleqa::tulu-thinker_deepseek,bbh:cot::hamish_zs_reasoning_deepseek_v2,gpqa:0shot_cot::hamish_zs_reasoning_deepseek,zebralogic::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,minerva_math::hamish_zs_reasoning_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,gsm8k::zs_cot_latex_deepseek,omega_500:0-shot-chat_deepseek,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek
