#!/usr/bin/env bash
#
# Download a model's weights into the shared weka HuggingFace cache so that a
# later serving job starts against a warm cache.
#
# Why this exists: Kimi-K3 is 1,561 GB across 96 shards and had never been
# pulled. Serving it directly meant eight B300s sat idle while vLLM downloaded,
# and because the job was unauthenticated the HF Hub rate-limited it so hard
# that four hours produced no shard-loading progress at all. Downloading on one
# GPU slot costs a fraction of that and leaves the cache warm for every later
# run of the same model.
#
# Env vars:
#   MODEL          HF repo id to download (required)
#   HF_CACHE_DIR   cache root; defaults to the same weka path the sweep uses
#   HF_TOKEN       set from a Beaker secret; anonymous pulls are rate-limited

set -uo pipefail

log() { printf '\n=== [%s] %s ===\n' "$(date -u +%H:%M:%S)" "$*"; }

: "${MODEL:?set MODEL}"

# Must match run_thinking_traces_sweep_in_job.sh exactly, or the serving job
# will not find what this job downloaded. BEAKER_USER_ID is unset in these jobs,
# so both fall through to "shared".
if [ -z "${HF_CACHE_DIR:-}" ] && [ -d /weka/oe-adapt-default ]; then
    HF_CACHE_DIR="/weka/oe-adapt-default/${BEAKER_USER_ID:-shared}/hf_cache"
fi
: "${HF_CACHE_DIR:=/tmp/hf_cache}"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

if [ -n "${HF_TOKEN:-}" ]; then
    log "HF_TOKEN present (${#HF_TOKEN} chars); authenticated download"
else
    log "WARNING: no HF_TOKEN. Anonymous downloads are rate-limited and a"
    log "  multi-hundred-GB checkpoint may never finish."
fi

log "model: ${MODEL}"
log "cache: ${HF_CACHE_DIR}"
df -h /weka/oe-adapt-default 2>/dev/null | tail -1 || true

# Report cache growth while the download runs, so progress is visible in the
# job log even though the downloader's own bars do not survive redirection.
( while true; do
    sleep 120
    log "cache size: $(du -sh "$HF_CACHE_DIR" 2>/dev/null | cut -f1) elapsed ${SECONDS}s"
  done ) &
watch_pid=$!

start=$SECONDS
uvx --python 3.12 --from "huggingface_hub[cli,hf_transfer]" \
    hf download "$MODEL" --max-workers 16
rc=$?
elapsed=$(( SECONDS - start ))

kill "$watch_pid" 2>/dev/null || true

if [ "$rc" -ne 0 ]; then
    log "FAILED: hf download exited ${rc} after ${elapsed}s"
    exit "$rc"
fi

log "DONE in ${elapsed}s"
log "final cache size: $(du -sh "$HF_CACHE_DIR" 2>/dev/null | cut -f1)"
find "$HF_CACHE_DIR" -name "*.safetensors*" -o -type f -size +1G 2>/dev/null | head -3
