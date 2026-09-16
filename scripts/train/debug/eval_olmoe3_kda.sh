#!/bin/bash
# Evaluate an OLMoE3 (KDA / MoE) HuggingFace checkpoint through olmo-eval.
#
#   ./scripts/train/debug/eval_olmoe3_kda.sh <hf_checkpoint_dir> <run_name> [tasks...]
#
# VERIFIED WORKING 2026-08-19 on Jacob's midtrain checkpoint: gsm8k 0.6308
# (Beaker 01M0BWD2Q61WVYYZRK1GEM6TA4).
#
# Every element below is load-bearing; this stack needs a specific torch/vLLM
# combination and olmo-eval's default image is a different one. Loosening pins
# does not work -- matching the runtime does. Derived from
# ladders/olmoe3/workloads/evals/eval.py in scaling-ladders (branch
# akshitab/emo_modularity), which is the source of truth: prefer running that
# launcher directly when you can, and read its whole _build_command before
# changing anything here.
#
# Requires an olmo-eval checkout at commit 97166c2 ("Exclude Torch companion
# packages from CUDA constraints"); earlier revisions symlink torch into the vLLM
# venv, so the runtime torch swap dies with "failed to remove .../torchgen".
set -euo pipefail

CKPT="${1:?usage: $0 <hf_checkpoint_dir> <run_name> [tasks...]}"
RUN_NAME="${2:?usage: $0 <hf_checkpoint_dir> <run_name> [tasks...]}"
shift 2
TASKS=("${@:-gsm8k}")

OLMO_EVAL_DIR="${OLMO_EVAL_DIR:-/root/repos/olmo-eval-launch}"
# Plugins are installed from Weka rather than the private scaling-ladders repo:
# eval jobs carry no GitHub credential. (Their launcher instead passes
# --secret-env <user>_GITHUB_TOKEN:GITHUB_TOKEN.)
PLUGIN_DIR="${PLUGIN_DIR:-/weka/oe-adapt-default/abhishekr/repos/scaling-ladders-emo/ladders/olmoe3}"

# olmo-core is installed --no-deps (its torch pin would fight olmo-eval's), so its
# runtime dependencies have to be listed explicitly; the ladders team gets them
# free from their olmo-core runtime image. fla is needed by the KDA layers.
OC_DEPS="cached-path>=1.7.2,dataclass-extensions>=0.3.0,bettermap,importlib_resources,safetensors,rich,pandas,flash-linear-attention==0.4.1"
DEPS="datasets==4.8.4,vllm==0.19.1,${OC_DEPS}"
DEPS="${DEPS},ai2-olmo-core[transformers] @ git+https://github.com/allenai/OLMo-core.git@f2cf93839 --no-deps"
DEPS="${DEPS},olmoe3-vllm-plugin @ file://${PLUGIN_DIR}/vllm_plugin"
DEPS="${DEPS},olmoe3-transformers-plugin @ file://${PLUGIN_DIR}/transformers_plugin --no-deps"

TASK_ARGS=()
for task in "${TASKS[@]}"; do TASK_ARGS+=(-t "$task"); done

cd "$OLMO_EVAL_DIR"
uv run olmo-eval beaker launch \
    -H default \
    -o provider.kind=vllm \
    -o provider.num_instances=1 \
    -o provider.package=wheel \
    -o provider.dependencies="[${DEPS}]" \
    -o provider.kwargs.enforce_eager=true \
    -o provider.kwargs.mamba_ssm_cache_dtype=float32 \
    -o provider.kwargs.language_model_only=true \
    -o provider.kwargs.attention_backend=FLASH_ATTN \
    -o provider.kwargs.enable_flashinfer_autotune=false \
    -n "$RUN_NAME" -m "$CKPT" "${TASK_ARGS[@]}" \
    -I akshitab/olmo-core-tch2110cu128-rma-2026-08-04 \
    --gpus 1 --retries 3 \
    -e PYTHONPATH=/gantry-runtime/src \
    -e TRITON_PTXAS_PATH=/opt/conda/bin/ptxas \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e OLMO_EVAL_RUNTIME_TORCH_VERSION=2.10.0+cu128 \
    -e OLMO_EVAL_RUNTIME_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
    -c ai2/ceres -w ai2/open-instruct-dev -B ai2/oe-other \
    -p urgent --no-follow -y
