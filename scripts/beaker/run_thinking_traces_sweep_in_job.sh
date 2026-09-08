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
#   MAX_MODEL_LEN      context (default 131072)
#   MAX_TOKENS         per-completion cap (default 128000)
#   NUM_PROMPTS        prompts per model (default 1000)
#   NUM_SAMPLES        completions per prompt (default 8)
#   CONCURRENCY        in-flight requests (default 256)
#   HF_REPO_ID         optional dataset repo to push each model's traces to
#   TRACE_STORE        weka dir for durable traces + resume markers
#   SYNC_INTERVAL      seconds between syncs (default 180)

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
: "${SERVE_PORT:=8008}"
: "${GPU_COUNT:=4}"
: "${TP_SIZE:=$GPU_COUNT}"
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
: "${VLLM_READY_TIMEOUT:=5400}"
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
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

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

log "sweep configuration"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
cat <<EOF
  models        : ${MODELS}
  vLLM          : ${VLLM_PKG_VERSION}   TP=${TP_SIZE} over ${GPU_COUNT} GPUs
  context       : max_model_len=${MAX_MODEL_LEN}  max_tokens=${MAX_TOKENS}
  sampling      : ${NUM_PROMPTS} prompts x ${NUM_SAMPLES} samples, T=${TEMPERATURE} top_p=${TOP_P} seed=${SEED}
  concurrency   : ${CONCURRENCY}
  trace store   : ${TRACE_STORE}
  JIT paths off : DEEP_GEMM=${VLLM_USE_DEEP_GEMM} MOE_DEEP_GEMM=${VLLM_MOE_USE_DEEP_GEMM} FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}
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
    if command -v nvcc >/dev/null 2>&1 && [ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]; then
        log "nvcc already present: $(nvcc --version 2>/dev/null | tail -1)"
        return 0
    fi
    local base="https://developer.download.nvidia.com/compute/cuda/redist"
    local dest=/opt/cuda-jit
    mkdir -p "$dest"
    # CUDA 13 splits the toolchain finely, and a partial install fails late and
    # confusingly: without cuda_crt the build dies on 'crt/host_defines.h: No
    # such file or directory', and without libnvvm nvcc has no cicc to run.
    #   cuda_nvcc  - nvcc, ptxas, cudafe++, fatbinary, nvlink
    #   cuda_crt   - the crt/ headers nvcc's generated host code includes
    #   cuda_cudart- vector_types.h and friends
    #   cuda_cccl  - CUB/Thrust headers the kernels use
    #   libnvvm    - cicc, the NVVM device-compiler frontend
    local comp name tmpd
    for comp in \
        "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-13.1.115-archive.tar.xz" \
        "cuda_crt/linux-x86_64/cuda_crt-linux-x86_64-13.1.115-archive.tar.xz" \
        "cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.1.80-archive.tar.xz" \
        "cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-13.1.115-archive.tar.xz" \
        "libnvvm/linux-x86_64/libnvvm-linux-x86_64-13.1.115-archive.tar.xz"
    do
        name="$(basename "$comp")"
        log "fetching $name"
        if ! curl -fsSL "$base/$comp" -o "/tmp/$name"; then
            log "WARNING: could not download $name; JIT kernels will fail"
            return 1
        fi
        tmpd="$(mktemp -d)"
        tar -xf "/tmp/$name" -C "$tmpd" && rm -f "/tmp/$name"
        # every archive is a single <component>-archive/ dir holding bin/ include/ lib/
        cp -a "$tmpd"/*/. "$dest"/ 2>/dev/null || true
        rm -rf "$tmpd"
    done
    export CUDA_HOME="$dest"
    export PATH="$dest/bin:$PATH"
    # Some build paths hardcode /usr/local/cuda rather than reading CUDA_HOME.
    mkdir -p /usr/local/cuda
    cp -asn "$dest"/. /usr/local/cuda/ 2>/dev/null || true
    if command -v nvcc >/dev/null 2>&1; then
        log "nvcc: $(nvcc --version 2>/dev/null | tail -1)"
        log "  CUDA_HOME=$CUDA_HOME  cicc=$([ -x "$dest/nvvm/bin/cicc" ] && echo yes || echo MISSING)" \
            "crt_headers=$([ -f "$dest/include/crt/host_defines.h" ] && echo yes || echo MISSING)"
        # Prove the toolchain works before vLLM depends on it.
        printf '__global__ void k(){}\nint main(){return 0;}\n' > /tmp/probe.cu
        if nvcc -arch=sm_103 -o /tmp/probe /tmp/probe.cu 2>/tmp/probe.err; then
            log "  nvcc sm_103 compile probe: OK"
        else
            log "  nvcc sm_103 compile probe FAILED:"; tail -5 /tmp/probe.err
        fi
    else
        log "WARNING: nvcc still not on PATH after install"
    fi
}
ensure_nvcc || log "continuing without nvcc; JIT-dependent kernels may fail"

if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

UV_RUN=( uv run --no-project --python 3.11
         --with datasets --with transformers --with openai --with numpy --with huggingface_hub )
log "pre-resolving client deps"
"${UV_RUN[@]}" python -c "import datasets, transformers, openai, numpy, huggingface_hub; print('client deps ready')" \
    || { log "FATAL: client dependency resolution failed"; exit 1; }

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
model_extra_args() {
    case "$1" in
        *Qwen3.5*)       echo "--language-model-only" ;;
        *GLM-5.2*)       echo "--kv-cache-dtype fp8" ;;
        *DeepSeek-V3.2*) echo "--tokenizer-mode deepseek_v32" ;;
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

    if [ -f "$done_marker" ]; then
        log "SKIP ${model}: already complete ($(wc -l < "$store" 2>/dev/null || echo 0) traces in $store)"
        cp "$store" "$traces" 2>/dev/null || true
        return 0
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
    local extra; extra="$(model_extra_args "$model")"
    [ -n "$extra" ] && log "recipe flags for ${served}: ${extra}"

    # shellcheck disable=SC2086  # $extra must word-split into separate flags
    uvx --python 3.12 "vllm==${VLLM_PKG_VERSION}" serve "$model" \
        ${extra} \
        --served-model-name "$served" \
        --port "$SERVE_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
        --enable-prefix-caching \
        --trust-remote-code \
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

    # stream partial traces out while generating
    ( while true; do sleep "$SYNC_INTERVAL"
        cp "$traces" "$store" 2>/dev/null || true
        [ -f "$traces" ] && log "sync: $(wc -l < "$traces") traces for ${served}"
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
