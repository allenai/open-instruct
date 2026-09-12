#!/usr/bin/env bash
#
# Inner script for the multi-model thinking-trace sweep. Serves each model in
# MODELS with vLLM in turn, generates traces against it, tears it down, and
# moves on. One Beaker job covers the whole slate.
#
# Why one job rather than one per model: nothing outside the job advances the
# sequence, so the sweep keeps running unattended. The costs that buys back:
#
#   * Per-model isolation. A model that fails to serve or generate is logged and
#     skipped; the sweep continues with the next one.
#   * Idempotent resume. A model whose traces are already complete in the weka
#     store is skipped outright, so an auto-resumed job picks up where the
#     previous attempt stopped instead of redoing finished work.
#   * Periodic sync. Traces stream to /results and weka while generating, so a
#     crash costs at most SYNC_INTERVAL of work rather than the whole model.
#
# Env vars (set by beaker_configs/launch_thinking_traces_sweep.sh):
#   MODELS             space-separated HF repo ids, served in order
#   VLLM_PKG_VERSION   default 0.23.0 (validated by tmax on this image + B300;
#                      registers all four architectures. Newer vLLM JIT-builds
#                      FlashInfer kernels with nvcc, which the image lacks.)
#   (was: needs >= 0.28 for GLM-5.2/Kimi-K2.x/
#                      DeepSeek-V3.2/Qwen3.5 architectures)
#   GPU_COUNT/TP_SIZE  GPUs and tensor-parallel size (default: 4 / GPU_COUNT)
#   DCP_SIZE           decode-context-parallel size. MLA models replicate their
#                      latent KV cache under TP; DCP shards it along sequence
#                      instead, which is worth several-fold concurrency.
#   MAX_MODEL_LEN      context (default 131072)
#   MAX_TOKENS         per-completion cap (default 128000)
#   NUM_PROMPTS        prompts per model (default 1000)
#   NUM_SAMPLES        completions per prompt (default 8)
#   CONCURRENCY        in-flight requests (default 256)
#   HF_REPO_ID         optional dataset repo to push each model's traces to
#   TRACE_STORE        weka dir for durable traces + resume markers
#   SYNC_INTERVAL      seconds between syncs (default 180)
#   RESET_TRACES       1 to discard stored traces for these models before running

set -uo pipefail   # NOT -e: a failing model must not kill the sweep

log() { printf '\n=== [%s] %s ===\n' "$(date -u +%H:%M:%S)" "$*"; }

# vLLM buries the actual cause far above the final traceback -- the API server's
# wrapper exception is what lands in a tail, while the engine/worker error that
# explains it scrolled past. Grep the whole log for exception lines so the job
# log alone is enough to diagnose a failure.
dump_vllm_failure() {
    local logfile="$1"
    log "root-cause candidates in $logfile:"
    grep -aoE "(TypeError|ImportError|ValueError|RuntimeError|AttributeError|ModuleNotFoundError|OSError|AssertionError|NotImplementedError|KeyError|CUDA error|out of memory)[^\"]{0,200}" \
        "$logfile" 2>/dev/null | sort -u | tail -20 || true
    log "last 80 lines of $logfile:"
    tail -80 "$logfile" 2>/dev/null || true
}

: "${MODELS:?set MODELS}"
: "${VLLM_PKG_VERSION:=0.28.0}"
: "${FLASHINFER_WHEELS:=1}"          # prebuilt kernels; set empty to disable
: "${FLASHINFER_VERSION:=0.6.16.post3}"
: "${SERVE_PORT:=8008}"
: "${GPU_COUNT:=4}"
: "${TP_SIZE:=$GPU_COUNT}"
# Data parallelism is the fallback capacity lever for models that cannot use
# DCP. Under plain TP an MLA-style latent KV cache is replicated on every
# rank, so concurrency at full context is a fraction of what the memory could
# hold; under DP each rank owns its own cache and --enable-expert-parallel
# shards the MoE experts so the weights still fit. TP_SIZE * DP_SIZE must
# equal GPU_COUNT.
: "${DP_SIZE:=}"
: "${ENABLE_EP:=}"
: "${MAX_MODEL_LEN:=131072}"
: "${MAX_TOKENS:=128000}"
: "${MAX_PROMPT_TOKENS:=1536}"
: "${NUM_PROMPTS:=1000}"
: "${NUM_SAMPLES:=8}"
: "${TEMPERATURE:=0.6}"
: "${TOP_P:=0.95}"
: "${SEED:=1234}"
: "${CONCURRENCY:=256}"
: "${VLLM_MAX_NUM_SEQS:=$CONCURRENCY}"
: "${VLLM_READY_TIMEOUT:=14400}"
# vLLM has its own, much shorter deadline for engine-core startup (default
# 600s) that is independent of the wait loop below. Under DP every rank runs
# its own engine core loading its own weights, so a 700+ GB checkpoint blows
# through 600s and the API server aborts with "Timed out waiting for engine
# core processes to start" even while the shard reads are progressing fine.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
: "${DATASET:=allenai/Dolci-Think-SFT-7B}"
: "${RESULTS_DIR:=/results}"
: "${SYNC_INTERVAL:=180}"
: "${HF_REPO_ID:=}"

# vLLM reserves VLLM_PORT as the base of its internal port range; leaking one in
# makes every parallel rank derive the same rendezvous port (EADDRINUSE).
unset VLLM_PORT

# DeepGEMM JIT-compiles FP8 kernels with nvcc, which is not in the Beaker CUDA
# image: every FP8 MoE here dies at engine init with
#   Assertion error (deepgemm .../jit/compiler.hpp): std::filesystem::exists(nvcc_path)
# vLLM 0.28 defaults both of these to on, so they must be turned off explicitly.
# The CUTLASS fallback kernels ship precompiled in the wheel. This mirrors the
# tmax recipe, which keeps DeepGEMM opt-in for the same reason.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
# FlashInfer JIT-builds its sampling kernels for sm_103 the same way, and hits
# the same missing compiler:
#   /usr/local/cuda/bin/nvcc: not found ... ninja: build stopped
# There is no CUDA 13 *dev* Beaker image to get nvcc from, so route sampling
# through vLLM's native PyTorch path instead.
# Previously disabled because its JIT needed nvcc. With flashinfer-cubin the
# sampler is prebuilt and free, so leave vLLM's default in place.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"

REPO_ROOT="$(pwd)"
mkdir -p "$RESULTS_DIR"

if [ -z "${TRACE_STORE:-}" ] && [ -d /weka/oe-adapt-default ]; then
    TRACE_STORE="/weka/oe-adapt-default/${BEAKER_USER_ID:-shared}/thinking_traces"
fi
: "${TRACE_STORE:=/tmp/thinking_traces}"
mkdir -p "$TRACE_STORE"

if [ -z "${HF_CACHE_DIR:-}" ] && [ -d /weka/oe-adapt-default ]; then
    HF_CACHE_DIR="/weka/oe-adapt-default/${BEAKER_USER_ID:-shared}/hf_cache"
fi
if [ -n "${HF_CACHE_DIR:-}" ]; then
    mkdir -p "$HF_CACHE_DIR"; export HF_HOME="$HF_CACHE_DIR"
fi

# Keep FlashInfer's compiled kernels on weka. They are JIT-built per (version,
# arch) and the container-local cache dies with the job, so without this every
# model in the slate pays the same multi-minute sm_103 build again.
if [ -d /weka/oe-adapt-default ]; then
    _fi_cache="/weka/oe-adapt-default/${BEAKER_USER_ID:-shared}/flashinfer_cache"
    mkdir -p "$_fi_cache" /root/.cache
    rm -rf /root/.cache/flashinfer 2>/dev/null || true
    ln -sfn "$_fi_cache" /root/.cache/flashinfer
    log "flashinfer JIT cache -> $_fi_cache"
fi

# Read throughput from the weka HF cache varies by roughly 8x across nodes on
# the same cluster -- GLM-5.2 loaded all 141 shards in 7 minutes on one node and
# managed 38 in 57 minutes on another, with prefetch enabled both times. That is
# a larger effect than most of the serving knobs, and it is invisible until a
# multi-hour load is already underway. Measure it in the first few seconds so a
# bad node is obvious in the log rather than inferred later from a stalled
# progress bar.
probe_weka_read() {
    local dir="${HF_CACHE_DIR:-}"
    [ -n "$dir" ] && [ -d "$dir" ] || return 0
    # HuggingFace stores cache entries as blobs/<sha> with NO extension; the
    # *.safetensors names under snapshots/ are symlinks, so matching on name and
    # size finds nothing. Search the blobs by size instead.
    local f
    f="$(find "$dir" -type f -size +1G -print -quit 2>/dev/null)" || true
    [ -n "$f" ] || { log "weka read probe: no cached blob >1G to sample yet"; return 0; }
    local start end mb
    # iflag=count_bytes reinterprets count in BYTES, so `count=2048` read 2048
    # bytes rather than 2048 MiB while the arithmetic below still divided 2048 MB
    # by the elapsed time -- inflating the result by ~10^6 (a run reported
    # "342202 MB/s") and making the slow-node warning unreachable. Neither flag
    # was doing anything useful here since there is no skip=, so drop both and
    # let count=2048 mean 2048 blocks of bs=1M.
    local bytes
    start=$(date +%s%N)
    bytes=$(dd if="$f" of=/dev/null bs=1M count=2048 2>&1 | awk '/bytes/{print $1; exit}')
    end=$(date +%s%N)
    [ -n "$bytes" ] || bytes=0
    mb=$(awk -v s="$start" -v e="$end" -v b="$bytes" \
        'BEGIN{d=(e-s)/1e9; if(d>0 && b>0) printf "%.0f", (b/1048576)/d; else print "?"}')
    log "weka read probe: ~${mb} MB/s ($(awk -v b="$bytes" 'BEGIN{printf "%.1f", b/1073741824}') GiB sampled)"
    if [ "$mb" != "?" ] && [ "$mb" -lt 300 ] 2>/dev/null; then
        log "  WARNING: slow node. A 700 GB checkpoint would take >40 min to load here."
        log "  Consider relaunching to land on a different node."
    fi
}

log "sweep configuration"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
cat <<EOF
  models        : ${MODELS}
  vLLM          : ${VLLM_PKG_VERSION}   TP=${TP_SIZE}${DCP_SIZE:+ DCP=${DCP_SIZE}}${DP_SIZE:+ DP=${DP_SIZE}}${ENABLE_EP:+ EP=on} over ${GPU_COUNT} GPUs
  context       : max_model_len=${MAX_MODEL_LEN}  max_tokens=${MAX_TOKENS}
  sampling      : ${NUM_PROMPTS} prompts x ${NUM_SAMPLES} samples, T=${TEMPERATURE} top_p=${TOP_P} seed=${SEED}
  concurrency   : ${CONCURRENCY}
  trace store   : ${TRACE_STORE}
  flashinfer    : prebuilt wheels=${FLASHINFER_WHEELS:-off} v${FLASHINFER_VERSION} sampler=${VLLM_USE_FLASHINFER_SAMPLER}
  JIT paths off : DEEP_GEMM=${VLLM_USE_DEEP_GEMM} MOE_DEEP_GEMM=${VLLM_MOE_USE_DEEP_GEMM}
                  (image has no nvcc, so every JIT path must fall back to precompiled kernels)
  HF_HOME       : ${HF_HOME:-<default>}
  hub repo      : ${HF_REPO_ID:-<none: push skipped>}
EOF

# --- CUDA compiler --------------------------------------------------------------
# The Beaker CUDA images ship the runtime but not nvcc, and there is no
# cuda13-dev image. Every recent vLLM JIT-builds *something* for sm_103 on
# Blackwell -- DeepGEMM FP8 GEMMs, FlashInfer sampling, FlashInfer FMHA, and (even
# with VLLM_USE_FLASHINFER_MOE_FP8=0, which is already the default) FlashInfer's
# trtllm fused-MoE kernels, which every FP8 MoE in this slate hits. Disabling
# them one at a time just surfaces the next one, so install the compiler instead.
#
# These are NVIDIA's own redistributables pinned to 13.1, matching the image's
# CUDA 13.1 runtime, so nothing is version-mixed. 32 MB total.
ensure_nvcc() {
    local dest=/opt/cuda-jit
    local base="https://developer.download.nvidia.com/compute/cuda/redist"
    local manifest="redistrib_${CUDA_REDIST_VERSION:-13.1.2}.json"
    local cache="${TOOLCHAIN_CACHE:-}"
    if [ -z "$cache" ] && [ -d /weka/oe-adapt-default ]; then
        cache="/weka/oe-adapt-default/${BEAKER_USER_ID:-shared}/cuda-jit-${CUDA_REDIST_VERSION:-13.1.2}.tar"
    fi
    mkdir -p "$dest"

    if [ -n "$cache" ] && [ -f "$cache" ]; then
        log "restoring CUDA toolkit from $cache"
        if tar -xf "$cache" -C "$dest" 2>/dev/null; then
            export_cuda_env
            verify_nvcc && return 0
            log "cached toolkit did not verify; rebuilding"
            rm -rf "${dest:?}"/* 2>/dev/null || true
        fi
    fi

    # Install the whole toolkit rather than hand-picking components. Picking them
    # one at a time cost a full weight-load cycle per missing header -- cuda_crt,
    # then libcublas, then libcurand -- because each is only reached once
    # FlashInfer's JIT gets far enough to include it. The excluded set is
    # profilers, debuggers, the driver and datacenter tooling: none of them
    # supply headers a kernel build includes.
    log "assembling CUDA toolkit from $manifest (~2 GB, cached afterwards)"
    local paths
    paths="$(curl -fsSL "$base/$manifest" | python3 -c '
import json,sys
DENY={"nsight_compute","nsight_systems","nvidia_driver","cuda_gdb","cuda_documentation",
      "collectx_bringup","mft","mft_autocomplete","mft_oem","fabricmanager","imex","nvlsm",
      "libnvidia_nscq","cuda_sanitizer_api","nvidia_fs","libnvsdm","cuda_nsight","cuda_compat",
      "libcufile","cuda_cupti","libnpp","libnvjpeg"}
d=json.load(sys.stdin)
for k,v in d.items():
    if isinstance(v,dict) and "linux-x86_64" in v and k not in DENY:
        print(v["linux-x86_64"]["relative_path"])
')" || { log "WARNING: could not read CUDA manifest"; return 1; }

    local n=0 comp name tmpd
    for comp in $paths; do
        name="$(basename "$comp")"
        if ! curl -fsSL "$base/$comp" -o "/tmp/$name"; then
            log "WARNING: download failed for $name"
            continue
        fi
        tmpd="$(mktemp -d)"
        tar -xf "/tmp/$name" -C "$tmpd" 2>/dev/null && cp -a "$tmpd"/*/. "$dest"/ 2>/dev/null
        rm -rf "$tmpd" "/tmp/$name"
        n=$(( n + 1 ))
    done
    log "installed $n CUDA components into $dest"

    export_cuda_env
    if [ -n "$cache" ]; then
        log "caching toolkit -> $cache"
        tar -cf "${cache}.tmp$$" -C "$dest" . 2>/dev/null && mv -f "${cache}.tmp$$" "$cache" \
            || { log "  (cache write failed; continuing)"; rm -f "${cache}.tmp$$"; }
    fi
    verify_nvcc
}

export_cuda_env() {
    local dest=/opt/cuda-jit
    # nvcc searches $CUDA_HOME/lib64 on x86_64; the archives use lib/.
    [ -d "$dest/lib" ] && [ ! -e "$dest/lib64" ] && ln -sfn "$dest/lib" "$dest/lib64"
    export CUDA_HOME="$dest" CUDA_PATH="$dest"
    export PATH="$dest/bin:$PATH"
    export LD_LIBRARY_PATH="$dest/lib:${LD_LIBRARY_PATH:-}"
    mkdir -p /usr/local/cuda && cp -asn "$dest"/. /usr/local/cuda/ 2>/dev/null || true
}

# Compile and link a kernel that includes the headers FlashInfer's trtllm
# kernels pull in. Each of these was previously discovered only when vLLM
# attempted a real JIT build, 15-50 minutes into a run.
verify_nvcc() {
    local dest=/opt/cuda-jit h missing=""
    command -v nvcc >/dev/null 2>&1 || { log "WARNING: nvcc not on PATH"; return 1; }
    for h in crt/host_defines.h cuda_runtime.h cublasLt.h cublas_v2.h curand_kernel.h \
             cuda_fp8.h cuda_bf16.h cooperative_groups.h; do
        [ -f "$dest/include/$h" ] || missing="$missing $h"
    done
    [ -x "$dest/nvvm/bin/cicc" ] || missing="$missing cicc"
    [ -f "$dest/lib/libcudart_static.a" ] || missing="$missing libcudart_static.a"
    [ -n "$missing" ] && log "  MISSING:$missing"
    cat > /tmp/probe.cu <<'CUEOF'
#include <cublasLt.h>
#include <curand_kernel.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
__global__ void k(){}
int main(){return 0;}
CUEOF
    if nvcc -arch=sm_103 -o /tmp/probe /tmp/probe.cu 2>/tmp/probe.err; then
        log "  nvcc sm_103 compile+link probe: OK$([ -n "$missing" ] && echo " (but headers missing:$missing)")"
        return 0
    fi
    log "  nvcc sm_103 probe FAILED:"; tail -6 /tmp/probe.err
    return 1
}
ensure_nvcc || log "continuing without nvcc; JIT-dependent kernels may fail"
probe_weka_read

if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

UV_RUN=( uv run --no-project --python 3.11
         --with datasets --with transformers --with openai --with numpy --with huggingface_hub
         --with tiktoken --with blobfile --with sentencepiece --with protobuf )
log "pre-resolving client deps"
"${UV_RUN[@]}" python -c "import datasets, transformers, openai, numpy, huggingface_hub; print('client deps ready')" \
    || { log "FATAL: client dependency resolution failed"; exit 1; }

# Load every model's tokenizer before serving any of them. generate_traces
# needs it to count tokens, but only runs once vLLM is up -- so a missing
# tokenizer dependency surfaces an hour into the job, after the weight load and
# kernel compilation, and takes the whole model with it. Kimi-K2.6 cost exactly
# that: it served correctly after 3147s and then died on a missing tiktoken.
# Fetching a tokenizer takes seconds, so check them all up front.
TOKENIZER_FAIL_FILE=/tmp/tokenizer_failures
preflight_tokenizers() {
    log "pre-flighting tokenizers: $MODELS"
    : > "$TOKENIZER_FAIL_FILE"
    MODELS="$MODELS" "${UV_RUN[@]}" python -c '
import os
import transformers
for m in os.environ.get("MODELS", "").split():
    try:
        t = transformers.AutoTokenizer.from_pretrained(m, trust_remote_code=True)
        print("TOKENIZER OK   %s (vocab %s)" % (m, getattr(t, "vocab_size", "?")))
    except Exception as exc:
        print("TOKENIZER FAIL %s -> %s: %s" % (m, type(exc).__name__, str(exc)[:220]))
' 2>&1 | tee /tmp/preflight.log
    grep -a "^TOKENIZER FAIL" /tmp/preflight.log | awk '{print $3}' >> "$TOKENIZER_FAIL_FILE" || true
    if [ -s "$TOKENIZER_FAIL_FILE" ]; then
        log "WARNING: tokenizer load failed for:$(tr '\n' ' ' < "$TOKENIZER_FAIL_FILE")"
    fi
}

preflight_tokenizers

# Per-model serve flags taken from the published vLLM recipes
# (recipes.vllm.ai/<org>/<model>.json). These are correctness/efficiency flags,
# not tuning: --language-model-only skips the vision tower on a multimodal
# checkpoint we only use as a text model, --tokenizer-mode selects the
# tokenizer DeepSeek V3.2 actually ships, and GLM's FP8 KV cache is the
# vendor's B300 configuration.
#
# Reasoning parsers are deliberately NOT enabled. They only change how the
# server splits the response, and leaving them off keeps the literal <think>
# tags in content so one parser handles every model uniformly. The client
# handles reasoning_content correctly either way.
# Hybrid think/non-think models need the toggle set explicitly. DeepSeek-V3.2's
# template defaults to NON-thinking -- it prefills a closing </think>, so the
# model answers directly and every trace comes back empty. Its kwarg is
# "thinking"; Qwen's "enable_thinking" is silently ignored here. Qwen3.5, Kimi
# and GLM all default to thinking ON, so they need nothing.
model_chat_template_kwargs() {
    case "$1" in
        *DeepSeek-V3.2*) echo '{"thinking": true}' ;;
        *)               echo "" ;;
    esac
}

model_extra_args() {
    case "$1" in
        *Qwen3.5*)       echo "--language-model-only" ;;
        *GLM-5.2*)       echo "--kv-cache-dtype fp8" ;;
        *DeepSeek-V3.2*) echo "--tokenizer-mode deepseek_v32" ;;
        # DeepSeek-V4-Flash settings come from vLLM 0.28.0's own eval config,
        # tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-confidence-TP4.yaml.
        # The repo ships no jinja chat template, so --tokenizer-mode deepseek_v4
        # is required for the server to render chat at all; the reasoning parser
        # makes vLLM return reasoning_content separately, which generate_traces.py
        # already prefers over splitting on </think>. indexer_kv_dtype=mxfp4
        # shrinks the DSA indexer cache, which our own notes flag as a +23%
        # KV surcharge on DeepSeek that the planner does not model.
        # The upstream config also enables dspark speculative decoding; we skip it
        # deliberately -- it points at a different checkpoint (…-DSpark) and uses
        # probabilistic draft sampling with adaptive verification, which could
        # perturb the output length distribution this study measures.
        *DeepSeek-V4*)   echo "--tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 --block-size 256 --attention_config.indexer_kv_dtype=mxfp4 --kv-cache-dtype fp8" ;;
        *)               echo "" ;;
    esac
}

# --- per-model run ------------------------------------------------------------
run_one_model() {
    local model="$1"
    local served; served="$(basename "$model" | tr '[:upper:]' '[:lower:]')"
    local traces="$RESULTS_DIR/traces_${served}.jsonl"
    local store="$TRACE_STORE/traces_${served}.jsonl"
    local done_marker="$store.done"

    # RESET_TRACES exists because resume is otherwise too eager: if a run
    # produced wrong-but-parseable traces (e.g. a hybrid model served in
    # non-thinking mode), resume would preserve them and only fill the gaps,
    # silently mixing two different generation configurations in one dataset.
    if [ "${RESET_TRACES:-0}" = "1" ]; then
        log "RESET_TRACES=1: discarding any stored traces for ${served}"
        rm -f "$store" "$done_marker" "$traces"
    fi

    if [ -f "$done_marker" ]; then
        log "SKIP ${model}: already complete ($(wc -l < "$store" 2>/dev/null || echo 0) traces in $store)"
        cp "$store" "$traces" 2>/dev/null || true
        return 0
    fi

    if [ -s "$TOKENIZER_FAIL_FILE" ] && grep -qxF "$model" "$TOKENIZER_FAIL_FILE"; then
        log "SKIP ${model}: tokenizer will not load, so serving it would waste the weight load"
        return 1
    fi

    log "MODEL ${model} -> served as ${served}"
    local vllm_log=/tmp/vllm_${served}.log

    # Pre-seed from traces an interrupted attempt already produced, so the run
    # continues instead of restarting. Preemption protection maxes out at 8h and
    # a model takes longer, so partial state is the expected case.
    local resume_flag=()
    if [ -s "$store" ]; then
        cp "$store" "$traces"
        resume_flag=(--resume)
        log "resuming ${served} from $(wc -l < "$store") existing traces"
    fi

    # --python 3.12 is load-bearing, not tidiness. uvx otherwise resolves 3.11,
    # where flashinfer's fd_exchange module fails to import with
    # "TypeError: type 'array.array' is not subscriptable" -- array.array only
    # became subscriptable in 3.12. That module is pulled in by the multi-GPU
    # all-reduce path, so it breaks every TP>1 serve while TP=1 works fine.
    local ctk; ctk="$(model_chat_template_kwargs "$model")"
    [ -n "$ctk" ] && log "chat_template_kwargs for ${served}: ${ctk}"
    local extra; extra="$(model_extra_args "$model")"
    [ -n "$extra" ] && log "recipe flags for ${served}: ${extra}"

    # shellcheck disable=SC2086  # $extra must word-split into separate flags
    # flashinfer-cubin and flashinfer-jit-cache carry prebuilt kernels. vLLM's
    # wheel deps pull in flashinfer-python ONLY -- jit-cache ships just in the
    # official Docker image -- so without these, FlashInfer compiles every kernel
    # from source with nvcc during vLLM's warmup run.
    #
    # Measured on one node, five sequential boots (exp 01M28P2590XZZDYT2SQ4G8DGX0):
    #   without: time-to-ready 1790s, warmup 1450s, 52 concurrent nvcc at peak,
    #            64 runtime HTTPS fetches to edge.urm.nvidia.com for cubins
    #   with:    time-to-ready  290s, warmup 11.5s, zero compiler processes,
    #            zero CDN fetches
    # Same attention/MoE backends selected either way, so nothing is degraded.
    # Neither wheel is on PyPI at this version; they come from flashinfer.ai.
    uvx --python 3.12 \
        ${FLASHINFER_WHEELS:+--with flashinfer-cubin==${FLASHINFER_VERSION} \
          --with flashinfer-jit-cache==${FLASHINFER_VERSION} \
          --index-strategy unsafe-best-match \
          --find-links https://flashinfer.ai/whl/flashinfer-cubin/ \
          --find-links https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/} \
        "vllm==${VLLM_PKG_VERSION}" serve "$model" \
        ${extra} \
        --served-model-name "$served" \
        --port "$SERVE_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
        --enable-prefix-caching \
        ${DCP_SIZE:+--decode-context-parallel-size "$DCP_SIZE"} \
        ${DP_SIZE:+--data-parallel-size "$DP_SIZE"} \
        ${ENABLE_EP:+--enable-expert-parallel} \
        --trust-remote-code \
        --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY:-prefetch}" \
        >"$vllm_log" 2>&1 &
    local vllm_pid=$!

    log "waiting for vllm on :$SERVE_PORT (up to ${VLLM_READY_TIMEOUT}s; weights are 400-760 GB on a cold cache)"
    local deadline=$(( SECONDS + VLLM_READY_TIMEOUT )) ticks=0
    until curl -sf "http://localhost:$SERVE_PORT/v1/models" >/dev/null 2>&1; do
        if ! kill -0 "$vllm_pid" 2>/dev/null; then
            log "FAILED ${model}: vllm died."; dump_vllm_failure "$vllm_log"
            cp "$vllm_log" "$RESULTS_DIR/" 2>/dev/null || true
            return 1
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            log "FAILED ${model}: vllm not ready in ${VLLM_READY_TIMEOUT}s."; dump_vllm_failure "$vllm_log"
            kill "$vllm_pid" 2>/dev/null; cp "$vllm_log" "$RESULTS_DIR/" 2>/dev/null || true
            return 1
        fi
        ticks=$(( ticks + 1 ))
        if [ $(( ticks % 12 )) = 0 ]; then
            log "still loading ${served} (${SECONDS}s):"; tail -3 "$vllm_log" 2>/dev/null || true
        fi
        sleep 10
    done
    log "vllm ready for ${served} after ${SECONDS}s"

    # Stream partial traces out, and surface the telemetry vLLM already emits.
    # Trace counts alone cannot distinguish "KV-bound" from "concurrency-starved"
    # from "the traces are simply long" -- but Running/Waiting/KV-usage can, and
    # vLLM logs them every few seconds to a file nobody was reading.
    ( while true; do sleep "$SYNC_INTERVAL"
        cp "$traces" "$store" 2>/dev/null || true
        [ -f "$traces" ] && log "sync: $(wc -l < "$traces") traces for ${served}"
        grep -aoE "Avg generation throughput:[^,]*|Running: [0-9]+ reqs|Waiting: [0-9]+ reqs|GPU KV cache usage: [0-9.]+%|Prefix cache hit rate: [0-9.]+%" \
            "$vllm_log" 2>/dev/null | tail -5 | paste -sd' ' - | sed 's/^/  vllm: /' || true
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
            --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', ' '{printf "gpu%s %s%% %sMiB %sW  ", $1,$2,$3,$4} END{print ""}' | sed 's/^/  util: /' || true
      done ) &
    local sync_pid=$!

    PYTHONPATH="$REPO_ROOT" "${UV_RUN[@]}" python scripts/thinking_traces/generate_traces.py \
        --model "$served" --tokenizer "$model" \
        --api-base "http://localhost:${SERVE_PORT}/v1" \
        --dataset "$DATASET" \
        --num-prompts "$NUM_PROMPTS" --num-samples "$NUM_SAMPLES" \
        --temperature "$TEMPERATURE" --top-p "$TOP_P" \
        --max-tokens "$MAX_TOKENS" --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
        --seed "$SEED" --concurrency "$CONCURRENCY" \
        ${ctk:+--chat-template-kwargs "$ctk"} \
        --prompts-output "$RESULTS_DIR/prompts_${served}.jsonl" \
        ${resume_flag[@]+"${resume_flag[@]}"} \
        --output "$traces" 2>&1 | tee -a "$RESULTS_DIR/generate_${served}.log"
    local rc=${PIPESTATUS[0]}

    kill "$sync_pid" 2>/dev/null; wait "$sync_pid" 2>/dev/null
    kill "$vllm_pid" 2>/dev/null; wait "$vllm_pid" 2>/dev/null
    tail -200 "$vllm_log" > "$RESULTS_DIR/vllm_tail_${served}.log" 2>/dev/null || true

    if [ "$rc" != "0" ]; then
        log "FAILED ${model}: generation exited $rc (partial traces kept)"
        cp "$traces" "$store" 2>/dev/null || true
        return 1
    fi

    cp "$traces" "$store" && touch "$done_marker"
    log "DONE ${model}: $(wc -l < "$traces") traces"

    PYTHONPATH="$REPO_ROOT" "${UV_RUN[@]}" python scripts/thinking_traces/analyze_traces.py \
        --traces "${served}=${traces}" \
        --json-output "$RESULTS_DIR/summary_${served}.json" 2>&1 \
        | tee "$RESULTS_DIR/summary_${served}.txt"

    if [ -n "$HF_REPO_ID" ]; then
        log "pushing ${served} to ${HF_REPO_ID} (best effort)"
        PYTHONPATH="$REPO_ROOT" "${UV_RUN[@]}" python scripts/thinking_traces/push_traces_to_hub.py \
            --traces "$traces" --repo-id "$HF_REPO_ID" --config-name "$served" --best-effort 2>&1 | tail -20
    fi
    return 0
}

# --- sweep --------------------------------------------------------------------
SUCCEEDED=(); FAILED=()
for model in $MODELS; do
    if run_one_model "$model"; then SUCCEEDED+=("$model"); else FAILED+=("$model"); fi
    log "progress: ${#SUCCEEDED[@]} succeeded, ${#FAILED[@]} failed, of $(echo $MODELS | wc -w)"
done

log "sweep complete"
echo "  succeeded: ${SUCCEEDED[*]:-none}"
echo "  failed   : ${FAILED[*]:-none}"
ls -la "$RESULTS_DIR"
[ ${#SUCCEEDED[@]} -gt 0 ] || exit 1
exit 0
