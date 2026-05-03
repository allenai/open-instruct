#!/bin/bash

EXP_NAME="${EXP_NAME:-qwen3_4b_base_dapo_rollout_probe}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

NUM_GPUS="${NUM_GPUS:-1}"
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-urgent}"
WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"
TRACE_DIR="${TRACE_DIR:-/weka/oe-adapt-default/tylerm/deletable_rollouts/${EXP_NAME}/${RUN_NAME}}"

if [[ $# -gt 0 ]]; then
  shift
fi

uv run python mason.py \
  --task_name "${EXP_NAME}" \
  --description "${RUN_NAME}" \
  --cluster "${CLUSTER}" \
  --workspace "${WORKSPACE}" \
  --priority "${PRIORITY}" \
  --pure_docker_mode \
  --no_auto_dataset_cache \
  --image "${BEAKER_IMAGE}" \
  --preemptible \
  --num_nodes 1 \
  --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  --gpus "${NUM_GPUS}" \
  --budget ai2/oe-adapt \
  -- \
uv run open_instruct/benchmark_generators.py \
  --run_name "${RUN_NAME}" \
  --exp_name "${EXP_NAME}" \
  --output_dir "${TRACE_DIR}" \
  --model_name_or_path "Qwen/Qwen3-4B-Base" \
  --tokenizer_name_or_path "Qwen/Qwen3-4B-Base" \
  --chat_template_name qwen_instruct_user_boxed_math \
  --dataset_mixer_list hamishivi/DAPO-Math-17k-Processed_filtered 1.0 \
  --dataset_mixer_list_splits train \
  --num_unique_prompts_rollout 64 \
  --vllm_num_engines 8 \
  --max_prompt_token_length 2048 \
  --response_length 8192 \
  --pack_length 10240 \
  --vllm_num_engines 1 \
  --vllm_tensor_parallel_size 1 \
  --vllm_top_p 1.0 \
  --temperature 1.0 \
  --apply_verifiable_reward true \
  --verification_reward 10.0 \
  --save_traces \
  --rollouts_save_path "${TRACE_DIR}" \
  --run_all_instances \
  --seed 1 "$@"
