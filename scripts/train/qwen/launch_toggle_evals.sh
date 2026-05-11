#!/bin/bash
set -euo pipefail

CKPT_BASE="/weka/oe-adapt-default/allennlp/deletable_checkpoint/finbarrt"

declare -a CONFIGS=(
    "p50_m10:${CKPT_BASE}/qwen3_4b_base_dapo_toggle_p50_m10__1__1778338269_checkpoints/step_1000"
    "p50_m50:${CKPT_BASE}/qwen3_4b_base_dapo_toggle_p50_m50__1__1778338336_checkpoints/step_1000"
    "p50_m100:${CKPT_BASE}/qwen3_4b_base_dapo_toggle_p50_m100__1__1778338287_checkpoints/step_1000"
    "p80_m10:${CKPT_BASE}/qwen3_4b_base_dapo_toggle_p80_m10__1__1778338353_checkpoints/step_1000"
    "p80_m100:${CKPT_BASE}/qwen3_4b_base_dapo_toggle_p80_m100__1__1778338363_checkpoints/step_1000"
)

for entry in "${CONFIGS[@]}"; do
    NAME="${entry%%:*}"
    MODEL="${entry##*:}"
    echo "=== Launching eval for ${NAME}: ${MODEL} ==="
    uv run olmo-eval beaker launch -y \
        -m "${MODEL}" \
        --harness default -A with_background=true -B ai2/oe-omai \
        -o provider.kind=vllm_server -o provider.max_model_len=32768 -o provider.trust_remote_code=true \
        -t aime_2025:pass_at_32 -o max_tokens=16384 \
        --gpus 1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -c h100 -w ai2/open-instruct-dev -p urgent
done
