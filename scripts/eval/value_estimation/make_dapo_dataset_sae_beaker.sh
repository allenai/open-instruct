#!/bin/bash
# Build the SAE-probed value-estimation dataset on Beaker.
# Probes are placed at SAE boundaries (tokens with prob < sae_threshold), downsampled
# to max_probes evenly-spaced entries — matching training-time boundary selection.
# Args: $1 = Beaker image (default: ${BEAKER_USER}/open-instruct-integration-test)
#       $2 = model name or path (default: Qwen/Qwen3-4B-Base)
#       $3 = chat template name (default: qwen_instruct_user_boxed_math; pass "builtin" for instruct models)
# Results are written to /output/dapo_math_100pairs_sae.parquet (Beaker result mount).
DDMM=$(date +"%d%m")
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
MODEL_NAME_OR_PATH="${2:-Qwen/Qwen3-4B-Base}"
CHAT_TEMPLATE_NAME="${3:-qwen_instruct_user_boxed_math}"

uv run python mason.py \
    --cluster ai2/jupiter \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --num_nodes 1 \
    --gpus 8 \
    --max_retries 0 \
    --no_auto_dataset_cache \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -- python -m open_instruct.value_estimation make_dataset \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --output_path /output/dapo_math_100pairs_sae.parquet \
    --dataset_name hamishivi/DAPO-Math-17k-Processed_filtered \
    --target_num_pairs 100 \
    --rollouts_per_prompt 8 \
    --continuations_per_probe 32 \
    --probe_mode sae \
    --sae_threshold 0.2 \
    --max_probes 16 \
    --max_response_length 8192 \
    --chat_template_name "${CHAT_TEMPLATE_NAME}"
