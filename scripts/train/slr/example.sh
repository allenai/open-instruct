#!/bin/bash
# Olmo 3 7B Think RL + SLR-Bench: Improve reasoning via structured curricula
# and verifiable rewards.
#
# This extends the standard Olmo 3 7B Think RL recipe by adding SLR-Bench —
# a scalable logical reasoning benchmark with deterministic symbolic evaluation.
# By training on both the existing Dolci RL data and SLR-Bench together, the
# model gets stronger reasoning capabilities through precise, partial-credit
# feedback across 20 difficulty levels.
#
# See docs/algorithms/slr_bench.md for details on the dataset and verifier.
#
# Prerequisites:
#   - SWI-Prolog (`swipl`) on $PATH (included in the Docker image).
#
# Usage:
#   ./scripts/train/build_image_and_launch.sh scripts/train/slr/example.sh

BEAKER_IMAGE=${1:-nathanl/open_instruct_auto}

python mason.py \
    --budget ai2/oe-adapt \
    --cluster ai2/jupiter \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --gpus 8 \
    --max_retries 0 \
    --env RAY_CGRAPH_get_timeout=300 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env HOSTED_VLLM_API_BASE=http://ceres-cs-aus-447.reviz.ai2.in:8001/v1 \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
        --exp_name slr_bench_7b_olmo3_thinker \
        --beta 0.0 \
        --num_samples_per_prompt_rollout 8 \
        --num_unique_prompts_rollout 64 \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --output_dir /output \
        --kl_estimator 2 \
        --dataset_mixer_list allenai/Dolci-Think-RL-7B 1.0 "AIML-TUDA/SLR-Bench:v1-All" 1.0 \
        --dataset_mixer_list_splits train train \
        --dataset_transform_fn slr_bench_prepare_v1 rlvr_tokenize_v1 rlvr_max_length_filter_v1 \
        --dataset_mixer_eval_list allenai/Dolci-Think-RL-7B 8 \
        --dataset_mixer_eval_list_splits train \
        --max_token_length 10240 \
        --max_prompt_token_length 2048 \
        --response_length 32768 \
        --pack_length 35840 \
        --model_name_or_path allenai/Olmo-3-7B-Think-DPO \
        --chat_template_name olmo_thinker \
        --stop_strings "[/RULE]" \
        --non_stop_penalty False \
        --mask_truncated_completions False \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes 10000000 \
        --deepspeed_stage 3 \
        --num_learners_per_node 8 8 \
        --vllm_num_engines 16 \
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
        --clip_higher 0.272 \
        --code_api_url https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program \
        --code_pass_rate_reward_threshold 0.99 \
        --oe_eval_max_length 32768 \
        --checkpoint_state_freq 100 \
        --backend_timeout 1200 \
        --inflight_updates true \
        --async_steps 8 \
        --advantage_normalization_type centered \
        --truncated_importance_sampling_ratio_cap 2.0
