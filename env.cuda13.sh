# Source this before working in the CUDA 13 worktree:  source env.cuda13.sh
# Isolates all CUDA/arch-specific caches from the shared weka ~/.cache used by the CUDA 12 box.
# (HuggingFace model cache is intentionally NOT overridden -- it stays shared & persistent.)
export UV_CACHE_DIR=/weka/nora-default/shashankg/.cache/uv-cuda13
export TRITON_CACHE_DIR=/weka/nora-default/shashankg/.cache/triton-cuda13
export TORCHINDUCTOR_CACHE_DIR=/weka/nora-default/shashankg/.cache/inductor-cuda13
export TORCH_EXTENSIONS_DIR=/weka/nora-default/shashankg/.cache/torch_ext-cuda13
# This container has no ssh binary, but the machine .gitconfig rewrites https->ssh for
# github/hf. Use a stripped global config (keeps identity+lfs, drops the ssh rewrite) so
# uv can clone public git deps (OLMo-core) over plain https.
export GIT_CONFIG_GLOBAL=/weka/nora-default/shashankg/code/open-instruct-cuda13/gitconfig.nossh
echo "[env.cuda13] isolated caches active; https git (no ssh); .venv is this worktree's own"
