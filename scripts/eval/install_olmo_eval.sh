#!/usr/bin/env bash
# Usage: install_olmo_eval.sh <git-ref>  (requires $GITHUB_TOKEN)
set -euo pipefail

ref="$1"

export UV_PROJECT_ENVIRONMENT=/opt/venv
export UV_CACHE_DIR=/weka/oe-eval-default/olmo-eval-pypi-cache

git clone "https://x-access-token:${GITHUB_TOKEN}@github.com/allenai/olmo-eval-internal.git" /opt/olmo-eval-internal
cd /opt/olmo-eval-internal
git checkout "$ref"

# Install vllm into a separate venv that symlinks the image's torch/nvidia libs,
# so we don't reinstall CUDA wheels. The main venv keeps the s3/clients/hf extras.
uv pip freeze -q | grep -E '^(torch|nvidia-)' > /tmp/cuda-constraints.txt
uv venv /opt/vllm-venv
for pkg in /opt/venv/lib/python*/site-packages/torch* /opt/venv/lib/python*/site-packages/nvidia*; do
    ln -sf "$pkg" /opt/vllm-venv/lib/python*/site-packages/
done
VIRTUAL_ENV=/opt/vllm-venv uv pip install --cache-dir "$UV_CACHE_DIR" -e '.[vllm]'
uv pip install -e '.[s3,clients,hf]' -c /tmp/cuda-constraints.txt
uv pip install 'antlr4-python3-runtime' -c /tmp/cuda-constraints.txt

cd /workspace
