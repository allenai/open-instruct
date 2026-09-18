#!/bin/bash
# Evaluate an Olmo 3.5 hero-small (OLMoE3 latent-KDA MoE) HuggingFace export through olmo-eval.
#
#   eval_olmoe3_hero.sh <hf_checkpoint_dir> <run_name> -t <task> [-o k=v ...] [-t <task> ...]
#
# Sibling of eval_olmoe3_kda.sh (the proxy launcher) for the hero checkpoints
# (hero-sft-anchor (#1895)). Same olmo-eval + vLLM stack, three changes, all
# forced by the hero architecture:
#
# * olmo-core is the hero lineage pin (89e7dcb7, the commit the SFT trains on):
#   the proxy pin f2cf93839 has no scalable_softmax / qk_norm_per_head_gains /
#   global_load_balancing, which layers 7 and 15 of these checkpoints use.
# * The vLLM/transformers plugins come from scaling-ladders' hero eval branch
#   (codex/small-hero-eval-20260909, staged on WEKA because eval jobs carry no
#   GitHub credential). That plugin pins flash-linear-attention==0.5.2.
# * OLMO_VLLM_TORCH_GROUPED_MOE=1 and OLMO_VLLM_FLA_KDA=1 select Jacob's "fast
#   BF16 grouped-MoE/FLA" inference profile, the one his hero base and SFT evals
#   use. His strict conversion/FP32 parity receipts cover a *precise* profile
#   (fp32, 8192 context, one sequence at a time) that cannot run 32K-cap
#   generation batteries; the fast profile is provisional in his words, so
#   record that caveat with any number produced here.
#
# Everything else is verbatim from eval_olmoe3_kda.sh / eval_paper.sh, which
# were verified on the proxy: torch 2.10 cu128 runtime swap on the olmo-core
# image, the flash-attn stub, the constraints file that keeps huggingface-hub
# and transformers where the 2026-09-01 runs had them.
set -euo pipefail

CKPT="${1:?usage: $0 <hf_checkpoint_dir> <run_name> -t task [-o k=v] ...}"
RUN_NAME="${2:?usage: $0 <hf_checkpoint_dir> <run_name> -t task [-o k=v] ...}"
shift 2
TASK_ARGS=("$@")

# Paper-protocol worktree (strip_thinking + olmo3adapt variants) by default; point
# at /root/repos/olmo-eval-launch or olmo-eval-moe for the dev battery.
OLMO_EVAL_DIR="${OLMO_EVAL_DIR:-/root/repos/olmo-eval-paper}"
PLUGIN_DIR="${PLUGIN_DIR:-/weka/oe-adapt-default/abhishekr/repos/scaling-ladders-hero/ladders/olmoe3}"
OLMO_CORE_REV="${OLMO_CORE_REV:-89e7dcb739e2168f5f1022c49f35eec2167383b8}"
HANDOFF=/weka/oe-adapt-default/abhishekr/handoff/xarch-evals
CONSTRAINTS="${CONSTRAINTS:-$HANDOFF/constraints-kda-eval.txt}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
CLUSTERS="${CLUSTERS:--c ai2/ceres -c ai2/jupiter -c ai2/saturn}"
WORKSPACE="${WORKSPACE:-ai2/open-instruct-dev}"
PRIORITY="${PRIORITY:-urgent}"
GPUS="${GPUS:-1}"
NUM_INSTANCES="${NUM_INSTANCES:-1}"  # vLLM instances; set with GPUS for wide suites
HARNESS="${HARNESS:-default}"
EVAL_IMAGE="${EVAL_IMAGE:-akshitab/olmo-core-tch2110cu128-rma-2026-08-04}"
# Empty PTXAS_PATH selects Triton's bundled compiler (sandbox image has no conda).
PTXAS_ARGS=()
if [[ -n "${PTXAS_PATH-/opt/conda/bin/ptxas}" ]]; then
    PTXAS_ARGS=(-e "TRITON_PTXAS_PATH=${PTXAS_PATH-/opt/conda/bin/ptxas}")
fi
TIMEOUT="${TIMEOUT:-24h}"  # Beaker job timeout; paper-protocol popqa/MATH at one instance need >24h

OC_DEPS="cached-path>=1.7.2,dataclass-extensions>=0.3.0,bettermap,importlib_resources,safetensors,rich,pandas,flash-linear-attention==0.5.2"
DEPS="datasets==4.8.4,vllm==0.19.1,huggingface-hub==1.16.1,${OC_DEPS}"
DEPS="${DEPS},ai2-olmo-core[transformers] @ git+https://github.com/allenai/OLMo-core.git@${OLMO_CORE_REV} --no-deps"
DEPS="${DEPS},olmoe3-vllm-plugin @ file://${PLUGIN_DIR}/vllm_plugin"
DEPS="${DEPS},olmoe3-transformers-plugin @ file://${PLUGIN_DIR}/transformers_plugin --no-deps"
DEPS="${DEPS},flash-attn @ file://${HANDOFF}/fa-stub/dist/flash_attn-99.0.0-py3-none-any.whl"
DEPS="${DEPS},transformers==5.14.1,huggingface-hub==1.16.1"  # last steps win; see constraints file

cd "$OLMO_EVAL_DIR"
# shellcheck disable=SC2086
uv run olmo-eval beaker launch \
    -H "$HARNESS" \
    -o provider.kind=vllm \
    -o provider.dtype=bfloat16 \
    -o provider.num_instances="$NUM_INSTANCES" \
    -o provider.package=wheel \
    -o provider.max_model_len="$MAX_MODEL_LEN" \
    -o provider.dependencies="[${DEPS}]" \
    -o provider.kwargs.enforce_eager=true \
    -o provider.kwargs.mamba_ssm_cache_dtype=float32 \
    -o provider.kwargs.language_model_only=true \
    -o provider.kwargs.attention_backend=FLASH_ATTN \
    -o provider.kwargs.enable_flashinfer_autotune=false \
    -o provider.kwargs.enable_prefix_caching=false \
    -n "$RUN_NAME" -m "$CKPT" "${TASK_ARGS[@]}" \
    -I "$EVAL_IMAGE" \
    --gpus "$GPUS" --retries 3 -T "$TIMEOUT" \
    -e 'UV_CACHE_DIR=/weka/oe-eval-default/olmo-eval-pypi-cache && rm -rf /opt/*/lib/python3*/site-packages/flash_attn' \
    -e UV_CONSTRAINT="$CONSTRAINTS" \
    -e PYTHONPATH=/gantry-runtime/src \
    "${PTXAS_ARGS[@]}" \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e OLMO_VLLM_TORCH_GROUPED_MOE=1 \
    -e OLMO_VLLM_FLA_KDA=1 \
    -e OLMO_EVAL_RUNTIME_TORCH_VERSION=2.10.0+cu128 \
    -e OLMO_EVAL_RUNTIME_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
    $CLUSTERS -w "$WORKSPACE" -B ai2/oe-other \
    -p "$PRIORITY" --no-follow -y
