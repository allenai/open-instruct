#!/bin/bash

# Optionally install a vLLM wheel compatible with the model and image, then run
# the evaluator. Qwen3.5 text-only checkpoints require vLLM >= 0.27.1.
set -euo pipefail

if [[ -n "${EVAL_VLLM_WHEEL_URL:-}" ]]; then
    UV_CACHE_DIR="${EVAL_VLLM_UV_CACHE_DIR:-/weka/oe-adapt-default/allennlp/.cache/uv-vllm-0271-cu129}" \
        uv pip install --python /stage/.venv/bin/python --upgrade \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        "$EVAL_VLLM_WHEEL_URL"
fi

# Old images may contain optional compiled extensions that uv does not upgrade
# with PyTorch. Remove them so vLLM uses its current JIT/runtime kernels.
for stale_package in flashinfer-cubin flash-attn; do
    if uv pip show --python /stage/.venv/bin/python "$stale_package" >/dev/null 2>&1; then
        uv pip uninstall --python /stage/.venv/bin/python "$stale_package"
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec /stage/.venv/bin/python "$SCRIPT_DIR/math_vllm.py" "$@"
