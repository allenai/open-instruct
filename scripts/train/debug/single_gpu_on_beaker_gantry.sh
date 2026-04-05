#!/usr/bin/env bash
# Single-GPU GRPO on Beaker via gantry run. Runs the same entrypoint as local debug:
#   scripts/train/debug/grpo_fast.sh (after Ray node setup).
#
# Prerequisites:
#   - Commit and push before running; gantry clones the repo at your pushed commit
#     (use --allow-dirty only if you intentionally accept divergence).
#   - Beaker workspace secrets: e.g. hf_token -> HF_TOKEN, wandb_api_key -> WANDB_API_KEY
#     (names must match your workspace; adjust --env-secret mappings below).
#   - For clusters without Weka (not in open_instruct.launch_utils.WEKA_CLUSTERS), remove
#     --weka=... and the HF_HOME / HF_DATASETS_CACHE / HF_HUB_CACHE env vars.

set -euo pipefail

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

echo "Using Beaker image: $BEAKER_IMAGE"

gantry run \
    -y \
    --workspace ai2/open-instruct-dev \
    --budget ai2/oe-adapt \
    --priority urgent \
    --cluster ai2/jupiter \
    --cluster ai2/saturn \
    --gpus 1 \
    --beaker-image "$BEAKER_IMAGE" \
    --no-host-networking \
    --description "Single GPU on Beaker test script (gantry)." \
    --task-timeout 15m \
    --retries 0 \
    --shared-memory 10.24gb \
    --weka=oe-adapt-default:/weka/oe-adapt-default \
    --weka=oe-training-default:/weka/oe-training-default \
    --env RAY_CGRAPH_get_timeout=300 \
    --env VLLM_DISABLE_COMPILE_CACHE=1 \
    --env NCCL_DEBUG=ERROR \
    --env VLLM_LOGGING_LEVEL=WARNING \
    --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env HF_HOME=/weka/oe-adapt-default/allennlp/.cache/huggingface \
    --env HF_DATASETS_CACHE=/weka/oe-adapt-default/allennlp/.cache/huggingface \
    --env HF_HUB_CACHE=/weka/oe-adapt-default/allennlp/.cache/hub \
    --env-secret HF_TOKEN=hf_token \
    --env-secret WANDB_API_KEY=wandb_api_key \
    --install "uv sync --frozen" \
    -- bash -c 'export REPO_PATH="$(pwd)" && source configs/beaker_configs/ray_node_setup.sh && bash scripts/train/debug/grpo_fast.sh --hf_entity allenai --wandb_entity ai2-llm'
