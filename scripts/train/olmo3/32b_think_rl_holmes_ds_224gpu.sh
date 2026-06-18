#!/bin/bash
# DeepSpeed Holmes large-scale run for Olmo 3 32B Think DPO RL.
# Mirrors the 28x8 OLMo-core Holmes launcher, but uses grpo_fast.py.

set -euo pipefail

BEAKER_IMAGE=${1:-nathanl/open_instruct_auto}
run_stamp=$(date -u +%Y%m%d_%H%M%S)
eager_suffix=""
if [[ "${VLLM_ENFORCE_EAGER:-false}" == "true" ]]; then
    eager_suffix="_eager"
fi
export exp_name=${EXP_NAME:-h224_32b_think_ds${eager_suffix}_${run_stamp}_${RANDOM}}
export data_mix="hamishivi/math_rlvr_mixture_dpo 1.0 hamishivi/code_rlvr_mixture_dpo 1.0 hamishivi/IF_multi_constraints_upto5_filtered_dpo_0625_filter 30186 allenai/rlvr_general_mix-keyword-filtered 21387"
export model_path=${MODEL_PATH:-allenai/Olmo-3-32B-Think-DPO}
export tokenizer_name_or_path=${TOKENIZER_NAME_OR_PATH:-allenai/Olmo-3-32B-Think-DPO}
export chat_template_name=${CHAT_TEMPLATE_NAME:-olmo_thinker}
export judge_base_url=${JUDGE_BASE_URL:?Set JUDGE_BASE_URL to the hosted Qwen3-32B judge endpoint, e.g. http://holmes-cs-aus-000.reviz.ai2.in:8001/v1}
export num_samples_per_prompt_rollout=${NUM_SAMPLES_PER_PROMPT_ROLLOUT:-8}
export num_unique_prompts_rollout=${NUM_UNIQUE_PROMPTS_ROLLOUT:-64}
export total_episodes=${TOTAL_EPISODES:-10000000}
export response_length=${RESPONSE_LENGTH:-32768}
export pack_length=${PACK_LENGTH:-$((response_length + 3072))}
export save_freq=${SAVE_FREQ:-25}
export checkpoint_state_freq=${CHECKPOINT_STATE_FREQ:-100}

vllm_enforce_eager_args=()
if [[ "${VLLM_ENFORCE_EAGER:-false}" == "true" ]]; then
    vllm_enforce_eager_args=(--vllm_enforce_eager)
fi
vllm_inference_batch_size_env=()
if [[ -n "${VLLM_INFERENCE_BATCH_SIZE:-}" ]]; then
    vllm_inference_batch_size_env=(--env OPEN_INSTRUCT_VLLM_INFERENCE_BATCH_SIZE=${VLLM_INFERENCE_BATCH_SIZE})
fi
vllm_compilation_config_env=()
if [[ -n "${VLLM_COMPILATION_CONFIG:-}" ]]; then
    vllm_compilation_config_env=(--env OPEN_INSTRUCT_VLLM_COMPILATION_CONFIG=${VLLM_COMPILATION_CONFIG})
fi
vllm_rms_norm_fake_patch_env=()
if [[ "${PATCH_VLLM_RMS_NORM_FAKE_IMPL:-false}" == "true" ]]; then
    vllm_rms_norm_fake_patch_env=(--env OPEN_INSTRUCT_PATCH_VLLM_RMS_NORM_FAKE_IMPL=1)
fi
disable_vllm_request_seed_env=()
if [[ "${DISABLE_VLLM_REQUEST_SEED:-false}" == "true" ]]; then
    disable_vllm_request_seed_env=(--env OPEN_INSTRUCT_DISABLE_VLLM_REQUEST_SEED=1)
fi

uv run python mason.py \
    --cluster ai2/holmes \
    --image $BEAKER_IMAGE \
    --pure_docker_mode \
    --workspace ai2/holmes-testing \
    --priority urgent \
    --preemptible \
    --num_nodes 28 \
    --gpus 8 \
    --max_retries 0 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ENGINE_READY_TIMEOUT_S=7200 \
    --env OPEN_INSTRUCT_VLLM_ENGINE_INIT_TIMEOUT_S=7500 \
    "${vllm_inference_batch_size_env[@]}" \
    "${vllm_compilation_config_env[@]}" \
    "${vllm_rms_norm_fake_patch_env[@]}" \
    "${disable_vllm_request_seed_env[@]}" \
    --env LD_LIBRARY_PATH=/var/lib/tcpxo/lib64 \
    --env NCCL_LIB_DIR=/var/lib/tcpxo/lib64 \
    --env HOSTED_VLLM_API_BASE=${judge_base_url} \
    --no_auto_dataset_cache \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
        --exp_name ${exp_name} \
        --beta 0.0 \
        --num_samples_per_prompt_rollout ${num_samples_per_prompt_rollout} \
        --num_unique_prompts_rollout ${num_unique_prompts_rollout} \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --learning_rate 2e-6 \
        --per_device_train_batch_size 1 \
        --output_dir /output \
        --kl_estimator 2 \
        --dataset_mixer_list ${data_mix} \
        --dataset_mixer_list_splits train \
        --max_prompt_token_length 2048 \
        --response_length ${response_length} \
        --pack_length ${pack_length} \
        --model_name_or_path ${model_path} \
        --tokenizer_name_or_path ${tokenizer_name_or_path} \
        --chat_template_name ${chat_template_name} \
        --non_stop_penalty False \
        --mask_truncated_completions False \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes ${total_episodes} \
        --deepspeed_stage 3 \
        --num_learners_per_node 8 8 8 8 8 8 8 8 8 8 8 8 \
        --vllm_num_engines 6 \
        --gather_whole_model False \
        --vllm_tensor_parallel_size 8 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --save_freq ${save_freq} \
        --gradient_checkpointing \
        --with_tracking \
        --llm_judge_model hosted_vllm/Qwen/Qwen3-32B \
        --llm_judge_timeout 600 \
        --llm_judge_max_tokens 2048 \
        --llm_judge_max_context_length 32768 \
        --code_api_url https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program \
        --code_pass_rate_reward_threshold 0.99 \
        --checkpoint_state_freq ${checkpoint_state_freq} \
        --backend_timeout 1200 \
        --active_sampling \
        --deepspeed_zpg 32 \
        --keep_last_n_checkpoints -1 \
        --no_push_to_hub \
        "${vllm_enforce_eager_args[@]}"
