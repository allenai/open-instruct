#!/bin/bash
set -euo pipefail

BEAKER_IMAGE=${1:-nathanl/open_instruct_auto}

# data mixes
nonreasoner_integration_mix_decon="hamishivi/rlvr_acecoder_filtered_filtered 20000 hamishivi/omega-combined-no-boxed_filtered 20000 hamishivi/rlvr_orz_math_57k_collected_filtered 14000 hamishivi/polaris_53k 14000 hamishivi/MathSub-30K_filtered 9000 hamishivi/DAPO-Math-17k-Processed_filtered 7000 allenai/IF_multi_constraints_upto5_filtered_dpo_0625_filter-keyword-filtered-topic-char-topic-filtered 38000 allenai/rlvr_general_mix-keyword-filtered-topic-chars-char-filt-topic-filtered 50000" #hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2_filtered 22000 hamishivi/virtuoussy_multi_subject_rlvr_filtered 20000 hamishivi/new-wildchat-english-general_filtered 19000"

# eval suite
general_evals_int="gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,omega_500:0-shot-chat_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,bbh:cot::hamish_zs_reasoning_deepseek_v2,mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek"

# model checkpoint
model_name_or_path="${MODEL_NAME_OR_PATH:-/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-instruct-dpo-1116-vibes/olmo3-7b-DPO-1115-newb-tpc-d5-lbc100-bal-1e-6-1__42__1763293644}" # nov 16 tentative final checkpoint row 33 # was replace by row 27

# cluster
cluster=ai2/jupiter
#template
chat_template=${CHAT_TEMPLATE_NAME:-olmo123} #olmo

NUM_NODES=${NUM_NODES:-1}
NUM_GPUS=${NUM_GPUS:-8}
NUM_LEARNERS_PER_NODE=${NUM_LEARNERS_PER_NODE:-6}
VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-2}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
NUM_UNIQUE_PROMPTS_ROLLOUT=${NUM_UNIQUE_PROMPTS_ROLLOUT:-12}
NUM_SAMPLES_PER_PROMPT_ROLLOUT=${NUM_SAMPLES_PER_PROMPT_ROLLOUT:-2}
NUM_MINI_BATCHES=${NUM_MINI_BATCHES:-2}
MAX_PROMPT_TOKEN_LENGTH=${MAX_PROMPT_TOKEN_LENGTH:-2048}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-2048}
PACK_LENGTH=${PACK_LENGTH:-4096}
hosted_vllm=""
gs_model_name="olmo3-instruct-dpo-hpz1"
exp_name="grpo_single_node_p${NUM_UNIQUE_PROMPTS_ROLLOUT}_${NUM_SAMPLES_PER_PROMPT_ROLLOUT}_${gs_model_name}"

EXP_NAME=${EXP_NAME:-${exp_name}}

required_gpus=$((NUM_LEARNERS_PER_NODE + VLLM_NUM_ENGINES * VLLM_TENSOR_PARALLEL_SIZE))
if ((NUM_NODES != 1)); then
    echo "This recipe is sized for one node; got NUM_NODES=${NUM_NODES}." >&2
    exit 2
fi
if ((required_gpus > NUM_GPUS)); then
    echo "Requested ${required_gpus} GPUs (${NUM_LEARNERS_PER_NODE} learners + ${VLLM_NUM_ENGINES} vLLM engines x TP ${VLLM_TENSOR_PARALLEL_SIZE}), but NUM_GPUS=${NUM_GPUS}." >&2
    exit 2
fi
rollout_batch_size=$((NUM_UNIQUE_PROMPTS_ROLLOUT * NUM_SAMPLES_PER_PROMPT_ROLLOUT))
if ((rollout_batch_size % NUM_LEARNERS_PER_NODE != 0)); then
    echo "Rollout batch ${rollout_batch_size} must be divisible by ${NUM_LEARNERS_PER_NODE} learners." >&2
    exit 2
fi
per_learner_batch_size=$((rollout_batch_size / NUM_LEARNERS_PER_NODE))
if ((per_learner_batch_size % NUM_MINI_BATCHES != 0)); then
    echo "Per-learner batch ${per_learner_batch_size} must be divisible by NUM_MINI_BATCHES=${NUM_MINI_BATCHES}." >&2
    exit 2
fi
if ((PACK_LENGTH < MAX_PROMPT_TOKEN_LENGTH + RESPONSE_LENGTH)); then
    echo "PACK_LENGTH=${PACK_LENGTH} must cover MAX_PROMPT_TOKEN_LENGTH + RESPONSE_LENGTH." >&2
    exit 2
fi

if [[ "${LOCAL_NPU_SMOKE:-0}" == "1" ]]; then
    : "${DATASET_PATH:?DATASET_PATH must point to a local dataset for LOCAL_NPU_SMOKE}"
    PYTHON_BIN=${PYTHON_BIN:-python}
    OUTPUT_DIR=${OUTPUT_DIR:-/tmp/open_instruct_olmo3_7b_instruct_rl_smoke}
    DATASET_SIZE=${DATASET_SIZE:-64}
    DATASET_LOCAL_CACHE_DIR=${DATASET_LOCAL_CACHE_DIR:-/tmp/open_instruct_dataset_cache}
    MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-1}
    TOTAL_EPISODES=${TOTAL_EPISODES:-${rollout_batch_size}}
    ASYNC_STEPS=${ASYNC_STEPS:-1}
    DEEPSPEED_STAGE=${DEEPSPEED_STAGE:-3}
    VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.45}
    ACTIVE_SAMPLING=${ACTIVE_SAMPLING:-0}

    if [[ "${ACTIVE_SAMPLING}" == "1" ]] && ((ASYNC_STEPS <= 1)); then
        echo "ACTIVE_SAMPLING=1 requires ASYNC_STEPS > 1." >&2
        exit 2
    fi

    mkdir -p "${OUTPUT_DIR}" "${DATASET_LOCAL_CACHE_DIR}"
    local_sampling_args=(--no_filter_zero_std_samples)
    if [[ "${ACTIVE_SAMPLING}" == "1" ]]; then
        local_sampling_args=(--active_sampling --no_resampling_pass_rate 0.875)
    fi

    if "${PYTHON_BIN}" open_instruct/grpo_fast.py \
        --exp_name "${EXP_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --beta 0.0 \
        --num_samples_per_prompt_rollout "${NUM_SAMPLES_PER_PROMPT_ROLLOUT}" \
        --num_unique_prompts_rollout "${NUM_UNIQUE_PROMPTS_ROLLOUT}" \
        --num_mini_batches "${NUM_MINI_BATCHES}" \
        --num_epochs 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --kl_estimator 2 \
        --dataset_mixer_list "${DATASET_PATH}" "${DATASET_SIZE}" \
        --dataset_mixer_list_splits train \
        --dataset_cache_mode local \
        --dataset_local_cache_dir "${DATASET_LOCAL_CACHE_DIR}" \
        --max_prompt_token_length "${MAX_PROMPT_TOKEN_LENGTH}" \
        --response_length "${RESPONSE_LENGTH}" \
        --pack_length "${PACK_LENGTH}" \
        --model_name_or_path "${model_name_or_path}" \
        --chat_template_name "${chat_template}" \
        --stop_strings "</answer>" \
        --non_stop_penalty False \
        --temperature 1.0 \
        --max_train_steps "${MAX_TRAIN_STEPS}" \
        --total_episodes "${TOTAL_EPISODES}" \
        --async_steps "${ASYNC_STEPS}" \
        --deepspeed_stage "${DEEPSPEED_STAGE}" \
        --num_learners_per_node "${NUM_LEARNERS_PER_NODE}" \
        --vllm_num_engines "${VLLM_NUM_ENGINES}" \
        --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --ground_truths_key ground_truth \
        --seed 1 \
        --local_eval_every -1 \
        --save_freq -1 \
        --checkpoint_state_freq -1 \
        --gradient_checkpointing \
        --vllm_enable_prefix_caching \
        --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
        --vllm_enforce_eager \
        --mask_truncated_completions False \
        --no_enable_queue_dashboard \
        --no_push_to_hub \
        --no_try_auto_save_to_beaker \
        "${local_sampling_args[@]}"
    then
        echo "RUNTIME_SELECTED_DEVICE=npu device=npu"
        echo "OLMO3_7B_INSTRUCT_RL_SMOKE=PASS"
        exit 0
    else
        exit $?
    fi
fi

uv run python mason.py \
        --description $exp_name \
        --task_name ${EXP_NAME} \
        --cluster ${cluster} \
        --workspace ai2/olmo-instruct  \
        --priority urgent \
        --pure_docker_mode \
        --image $BEAKER_IMAGE \
        --preemptible \
        --num_nodes ${NUM_NODES} \
        --max_retries 5 \
        --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        --env HOSTED_VLLM_API_BASE=$hosted_vllm \
        --gs_model_name $gs_model_name \
        --gpus ${NUM_GPUS} \
        --budget ai2/oe-other -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
        --exp_name ${EXP_NAME} \
        --beta 0.0 \
        --num_samples_per_prompt_rollout ${NUM_SAMPLES_PER_PROMPT_ROLLOUT} \
        --num_unique_prompts_rollout ${NUM_UNIQUE_PROMPTS_ROLLOUT} \
        --num_mini_batches ${NUM_MINI_BATCHES} \
        --num_epochs 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --kl_estimator 2 \
        --dataset_mixer_list ${nonreasoner_integration_mix_decon} \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list hamishivi/omega-combined 4 allenai/IF_multi_constraints_upto5 4 saurabh5/rlvr_acecoder_filtered 4 hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2 4 hamishivi/virtuoussy_multi_subject_rlvr 4 \
        --dataset_mixer_eval_list_splits train \
        --max_prompt_token_length ${MAX_PROMPT_TOKEN_LENGTH} --response_length ${RESPONSE_LENGTH} --pack_length ${PACK_LENGTH} \
        --model_name_or_path ${model_name_or_path} \
        --chat_template_name ${chat_template} \
        --stop_strings "</answer>" \
        --non_stop_penalty False \
        --temperature 1.0 \
        --total_episodes 1024000 \
        --deepspeed_stage 3 \
        --num_learners_per_node ${NUM_LEARNERS_PER_NODE} \
        --vllm_num_engines ${VLLM_NUM_ENGINES} \
        --vllm_tensor_parallel_size ${VLLM_TENSOR_PARALLEL_SIZE} \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --local_eval_every 50 \
        --save_freq 50 \
        --checkpoint_state_freq 50 \
        --beaker_eval_freq 50 \
        --gradient_checkpointing \
        --with_tracking \
        --vllm_enable_prefix_caching \
        --mask_truncated_completions False \
        --llm_judge_model hosted_vllm/Qwen/Qwen3-32B \
        --llm_judge_timeout 600 \
        --llm_judge_max_tokens 2048 \
        --llm_judge_max_context_length 32768 \
        --oe_eval_max_length 32768 \
        --try_launch_beaker_eval_jobs_on_weka True \
        --oe_eval_tasks ${general_evals_int} \
        --eval_priority urgent \
        --code_pass_rate_reward_threshold 0.99 \
        --active_sampling \
        --no_resampling_pass_rate 0.875 \
