#!/bin/bash
# Bootstrap open-instruct's vision branch inside a foreign CUDA-13 image on ai2/holmes.
#
# ai2/holmes runs B300s, which need the CUDA-13 dependency group; we have no CUDA-13
# image of this branch, so (following the olmo-miles pattern in this workspace) the
# job clones the branch at container start and `uv sync`s the cuda13 group from the
# lockfile. The uv cache lives on weka so restarts are fast.
#
# Usage (as the mason payload): bash <(curl ...) is NOT used; the launch script clones
# the repo first and then invokes this script with the training args:
#   holmes_bootstrap.sh <git-ref> <torchrun args...>
set -euo pipefail

GIT_REF="${1:?usage: holmes_bootstrap.sh <git-ref> <torchrun args...>}"
shift

# The bootstrap image may set PYTHONPATH to its own packages (olmo-miles ships an
# older olmo_core without data.multimodal that would shadow our venv's).
unset PYTHONPATH

export UV_CACHE_DIR=/weka/oe-adapt-default/allennlp/deletable_uv_cache_cuda13
export HF_HOME=/weka/oe-adapt-default/allennlp/deletable_hf_cache
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Syncing cuda13 environment for open-instruct@${GIT_REF} ..."
uv sync --frozen --no-group cuda12 --group cuda13
echo "Environment ready; launching:"
echo "torchrun $*"
exec uv run --no-sync torchrun "$@"
