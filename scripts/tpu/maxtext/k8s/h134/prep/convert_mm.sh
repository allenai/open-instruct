#!/bin/bash
# H134 B1/B2 (tpu-worker fork, 2026-10-26): strip the vision parts from a multimodal OLMo-core hero-small checkpoint
# and dump its LM masters for the unchanged text converter. One base per call:
#   convert_mm.sh <name> <checkpoint step dir> <work dir> <stage dir> [lm prefix]
# Stage 0  strip_vision.py inspect + plan (DCP metadata only; fails closed, see strip_vision.py)
# Stage 1  dump_olmocore_masters_mm.py: only the planned LM masters, one key at a time   -> <stage>/<name>-masters-mm-raw-h134
# Stage 1b strip_vision.py apply: B0 names, vocab rows sliced if appended, byte-checked  -> <stage>/<name>-masters-stripped-h134
# Stage 2 (not here; needs the MaxText env): convert_olmocore_to_maxtext.py <stripped dir> <stripped dir> <items dir>.
#   The stripped dir holds a config.json with the LM section and vocab_size = B0's 100352, which stage 2 reads.
# Writes only new paths; deletes nothing. Needs PY (python with torch + numpy) and OI (this repo's olmo35 dir).
set -euo pipefail
name=$1 src=$2 W=$3 STAGE=$4 prefix=${5:-}
: "${PY:?}" "${OI:?}" "${REF:?}" "${REFCFG:?}"
mkdir -p "$W"
log() { echo "$(date -u +%FT%TZ) [$name] $*" | tee -a "$W/convert_mm.log"; }
"$PY" "$OI/strip_vision.py" inspect "$src" "$W/inspect.json" | tee -a "$W/convert_mm.log"
tok=$("$PY" - "$src/config.json" <<'PY'
import json, sys
def find(d):
    if isinstance(d, dict):
        t = d.get("tokenizer")
        if isinstance(t, dict) and "vocab_size" in t:
            return t["vocab_size"]
        for v in d.values():
            r = find(v)
            if r is not None:
                return r
print(find(json.load(open(sys.argv[1]))) or "")
PY
)
log "tokenizer vocab_size from config: ${tok:-<none>}"
if [ -z "$prefix" ]; then  # LM prefix from the embedding key (module.<prefix>embeddings.weight.main), if any
  prefix=$("$PY" -c "import json,sys;m=json.load(open(sys.argv[1]))['main'];p=[k[len('module.'):-len('embeddings.weight.main')] for k in m if k.endswith('embeddings.weight.main')];print(p[0] if len(p)==1 else '')" "$W/inspect.json")
  log "LM prefix detected: '${prefix}'"
fi
args=(--ref "$REF" --ref-config "$REFCFG" --src "$src" --out "$W/plan.json")
[ -n "$prefix" ] && args+=(--lm-prefix "$prefix")
[ -n "$tok" ] && args+=(--mm-tokenizer-size "$tok")
"$PY" "$OI/strip_vision.py" plan "${args[@]}" >> "$W/convert_mm.log" 2>&1 || { log "STAGE0-FAIL: $(tail -1 "$W/convert_mm.log")"; exit 3; }
log "$(grep PLAN-OK "$W/convert_mm.log" | tail -1)"
[ "${STOP_AFTER_PLAN:-}" = 1 ] && { log "plan only"; exit 0; }
raw=$STAGE/$name-masters-mm-raw-h134 out=$STAGE/$name-masters-stripped-h134
ps=$(sha256sum "$W/plan.json" | cut -d' ' -f1)
if [ -e "$raw/DONE" ]; then [ "$(cat "$raw/DONE")" = "$ps" ] || { log "$raw was dumped for another plan; not reusing"; exit 4; }
else log "stage 1 start"; "$PY" "$OI/dump_olmocore_masters_mm.py" "$src" "$W/plan.json" "$raw" | tee -a "$W/convert_mm.log"; echo "$ps" > "$raw/DONE"; fi
log "stage 1b start"
"$PY" "$OI/strip_vision.py" apply --plan "$W/plan.json" --masters-in "$raw" --masters-out "$out" --receipt "$W/strip_receipt.json" | tee -a "$W/convert_mm.log"
log "CONVERT-MM-DONE $out"
