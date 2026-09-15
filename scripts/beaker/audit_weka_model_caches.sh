#!/usr/bin/env bash
#
# Find every HuggingFace-style model cache under a weka mount and report what
# each one holds, so duplicated checkpoints are visible.
#
# The sweep writes to /weka/oe-adapt-default/${BEAKER_USER_ID:-shared}/hf_cache,
# but nothing stops other jobs from caching the same weights elsewhere on the
# same volume, and these checkpoints are 150 GB to 1.5 TB each.

set -uo pipefail
log() { printf '\n=== [%s] %s ===\n' "$(date -u +%H:%M:%S)" "$*"; }

: "${ROOT:=/weka/oe-adapt-default}"
: "${MAXDEPTH:=7}"

log "volume"
df -h "$ROOT" 2>/dev/null | tail -1

log "scanning ${ROOT} (maxdepth ${MAXDEPTH}) for HF cache directories"
# HF stores each repo as models--<org>--<name>; finding those finds every cache
# root regardless of what the containing directory is called.
mapfile -t dirs < <(find "$ROOT" -maxdepth "$MAXDEPTH" -type d -name 'models--*' 2>/dev/null | sort)
log "found ${#dirs[@]} cached repos"

printf '\n%-14s  %s\n' "SIZE" "PATH"
for d in "${dirs[@]}"; do
    printf '%-14s  %s\n' "$(du -sh "$d" 2>/dev/null | cut -f1)" "$d"
done

log "totals by cache root"
for d in "${dirs[@]}"; do
    # strip the trailing /hub/models--... to get the cache root
    printf '%s\n' "${d%/hub/models--*}"
done | sort | uniq -c | sort -rn

log "duplicate repos (same model cached in more than one root)"
for d in "${dirs[@]}"; do
    printf '%s\n' "$(basename "$d")"
done | sort | uniq -d | while read -r repo; do
    echo "  ${repo}:"
    printf '%s\n' "${dirs[@]}" | grep -a "/${repo}\$" | sed 's/^/    /'
done

log "other large directories directly under ${ROOT} (top 15)"
du -sh "$ROOT"/* 2>/dev/null | sort -rh | head -15
log "done"
