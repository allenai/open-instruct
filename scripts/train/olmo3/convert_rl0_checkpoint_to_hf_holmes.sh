#!/bin/bash
# Convert a 32B RL0 OLMo-core checkpoint-state step to HuggingFace format on Holmes.

set -euo pipefail

BEAKER_IMAGE=${1:-nathanl/open_instruct_auto}
STEP=${STEP:-step1000}
SOURCE_RUN=${SOURCE_RUN:-/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/jacobm/1781769360_491181}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${SOURCE_RUN}/${STEP}/model_and_optim}
OUTPUT_DIR=${OUTPUT_DIR:-/weka/oe-adapt-default/allennlp/deletable_checkpoint/jacobm/rl0_32b_hf_checkpoints/${STEP}}
HF_MODEL_NAME=${HF_MODEL_NAME:-allenai/Olmo-3-1125-32B}
OLMO_CORE_MODEL_NAME=${OLMO_CORE_MODEL_NAME:-olmo3_32B}
TOKENIZER_NAME=${TOKENIZER_NAME:-${HF_MODEL_NAME}}
INIT_DEVICE=${INIT_DEVICE:-cpu}
WORK_DIR=${WORK_DIR:-/tmp/olmo_core_to_hf_${STEP}}

uv run python mason.py \
    --cluster ai2/holmes \
    --image "${BEAKER_IMAGE}" \
    --pure_docker_mode \
    --workspace ai2/holmes-testing \
    --priority urgent \
    --num_nodes 1 \
    --gpus 8 \
    --shared_memory 64gb \
    --timeout 12h \
    --max_retries 0 \
    --no_auto_dataset_cache \
    -- python scripts/train/convert_olmo_core_to_hf.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --hf-model-name "${HF_MODEL_NAME}" \
        --olmo-core-model-name "${OLMO_CORE_MODEL_NAME}" \
        --tokenizer-name "${TOKENIZER_NAME}" \
        --output-dir "${OUTPUT_DIR}" \
        --work-dir "${WORK_DIR}" \
        --init-device "${INIT_DEVICE}"
