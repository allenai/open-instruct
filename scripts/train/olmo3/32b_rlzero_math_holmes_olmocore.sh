#!/bin/bash
# OLMo-core/FSDP Holmes launcher for 32B RL-zero math.
# Defaults to a 9-node debug run; override NUM_NODES and related env vars when scaling.

set -euo pipefail

if [[ $# -gt 0 ]]; then
    BEAKER_IMAGE=$1
    shift
else
    BEAKER_IMAGE=nathanl/open_instruct_auto
fi
run_stamp=$(date -u +%Y%m%d_%H%M%S)

export exp_name=${EXP_NAME:-holmes_debug_olmo3_32b_rlzero_math_olmocore_${run_stamp}_${RANDOM}}
export model_name_or_path=${MODEL_NAME_OR_PATH:-allenai/Olmo-3-1125-32B}
export tokenizer_name_or_path=${TOKENIZER_NAME_OR_PATH:-${model_name_or_path}}
export hf_model_name_or_path=${HF_MODEL_NAME_OR_PATH:-${model_name_or_path}}
export data_mix=${DATA_MIX:-"allenai/Dolci-RLZero-Math-7B 1.0"}
export local_evals=${LOCAL_EVALS:-"allenai/aime_2025_openinstruct 1.0"}
export local_eval_splits=${LOCAL_EVAL_SPLITS:-train}
export oe_eval_tasks=${OE_EVAL_TASKS:-"aime:zs_cot_r1::pass_at_32_2024_rlzero,aime:zs_cot_r1::pass_at_32_2025_rlzero"}

export num_nodes=${NUM_NODES:-9}
export num_learners_per_node=${NUM_LEARNERS_PER_NODE:-"8 8 8 8"}
export fsdp_shard_degree=${FSDP_SHARD_DEGREE:-32}
export fsdp_num_replicas=${FSDP_NUM_REPLICAS:-1}
export activation_memory_budget=${ACTIVATION_MEMORY_BUDGET:-0.3}
export vllm_num_engines=${VLLM_NUM_ENGINES:-10}
export vllm_tensor_parallel_size=${VLLM_TENSOR_PARALLEL_SIZE:-4}
export num_samples_per_prompt_rollout=${NUM_SAMPLES_PER_PROMPT_ROLLOUT:-8}
export num_unique_prompts_rollout=${NUM_UNIQUE_PROMPTS_ROLLOUT:-32}
export response_length=${RESPONSE_LENGTH:-16384}
export pack_length=${PACK_LENGTH:-$((response_length + 2048))}
export total_episodes=${TOTAL_EPISODES:-768000}
export save_freq=${SAVE_FREQ:-100}
export checkpoint_state_freq=${CHECKPOINT_STATE_FREQ:-100}
export local_eval_every=${LOCAL_EVAL_EVERY:-100}

vllm_enforce_eager_args=()
if [[ "${VLLM_ENFORCE_EAGER:-false}" == "true" ]]; then
    vllm_enforce_eager_args=(--vllm_enforce_eager)
fi

uv run python mason.py \
    --task_name ${exp_name} \
    --cluster ai2/holmes \
    --workspace ai2/holmes-testing \
    --priority urgent \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes ${num_nodes} \
    --gpus 8 \
    --max_retries 0 \
    --no_auto_dataset_cache \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ENGINE_READY_TIMEOUT_S=7200 \
    --env OPEN_INSTRUCT_VLLM_ENGINE_INIT_TIMEOUT_S=7500 \
    --env OPEN_INSTRUCT_PATCH_OLMO_CORE_FLASH_ATTN_4_API=1 \
    --env LD_LIBRARY_PATH=/var/lib/tcpxo/lib64 \
    --env NCCL_LIB_DIR=/var/lib/tcpxo/lib64 \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo.py \
        --exp_name ${exp_name} \
        --beta 0.0 \
        --no_resampling_pass_rate 0.875 \
        --active_sampling \
        --num_samples_per_prompt_rollout ${num_samples_per_prompt_rollout} \
        --num_unique_prompts_rollout ${num_unique_prompts_rollout} \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --output_dir /output \
        --kl_estimator 2 \
        --dataset_mixer_list ${data_mix} \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list ${local_evals} \
        --dataset_mixer_eval_list_splits ${local_eval_splits} \
        --max_prompt_token_length 2048 \
        --response_length ${response_length} \
        --pack_length ${pack_length} \
        --model_name_or_path ${model_name_or_path} \
        --tokenizer_name_or_path ${tokenizer_name_or_path} \
        --hf_model_name_or_path ${hf_model_name_or_path} \
        --config_name olmo3_32B \
        --chat_template_name olmo_thinker_rlzero \
        --non_stop_penalty False \
        --temperature 1.0 \
        --total_episodes ${total_episodes} \
        --num_learners_per_node ${num_learners_per_node} \
        --fsdp_shard_degree ${fsdp_shard_degree} \
        --fsdp_num_replicas ${fsdp_num_replicas} \
        --activation_memory_budget ${activation_memory_budget} \
        --vllm_num_engines ${vllm_num_engines} \
        --vllm_tensor_parallel_size ${vllm_tensor_parallel_size} \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --local_eval_every ${local_eval_every} \
        --save_freq ${save_freq} \
        --checkpoint_state_freq ${checkpoint_state_freq} \
        --keep_last_n_checkpoints -1 \
        --gradient_checkpointing \
        --with_tracking \
        --vllm_enable_prefix_caching \
        --mask_truncated_completions False \
        --oe_eval_max_length 32768 \
        --try_launch_beaker_eval_jobs_on_weka True \
        --eval_priority urgent \
        --oe_eval_tasks ${oe_eval_tasks} \
        --oe_eval_gpu_multiplier 4 \
        --backend_timeout 1200 \
        --disable_async_bookkeeping \
        --bookkeeping_soft_timeout 3600 \
        "${vllm_enforce_eager_args[@]}" \
        "$@"
