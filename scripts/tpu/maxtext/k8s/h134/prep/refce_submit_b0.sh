#!/bin/bash
# H134 B0 (bf16): once convert_bf16.sh logs BF16-CONVERT-DONE, submit the HF step-1 CE job once; log the CEs when both land.
W=/weka/oe-eval-default/abhishekr/watch/tpu/h134; R=$W/refce; L=$W/convert-bf16/convert.log; O=$R/b0; mkdir -p $O
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $R/refce_b0.log >> $W/events.log; }
until grep -q 'BF16-CONVERT-DONE' $L 2>/dev/null; do grep -qE '(FAIL|MISMATCH)' $L 2>/dev/null && { log "b0 bf16 convert failed; no refce submitted"; exit 1; }; sleep 60; done
[ -f $O/experiment.id ] || { id=$(beaker experiment create --workspace ai2/olmo-instruct $R/beaker-refce-b0.yaml --format json 2>>$O/submit.err | python3 -c "import json,sys;d=json.load(sys.stdin);print((d[0] if isinstance(d,list) else d)['id'])")
  [ -n "$id" ] || { log "b0 refce submit failed (see refce/b0/submit.err)"; exit 2; }; echo $id > $O/experiment.id
  log "b0 refce submitted: Beaker $id (ai2/olmo-instruct, jupiter or holmes, 1 GPU, high, minRuntime 5m, timeout 4h)"; }
until [ -f $O/refce_c11-s0.json ] && [ -f $O/refce_c11-s1.json ]; do sleep 120; done
for s in 0 1; do log "REFCE b0 c11-s$s: $(python3 -c "import json;d=json.load(open('$O/refce_c11-s$s.json'));print('ce', round(d['ce'],6), 'loss_tokens', d['loss_tokens'])")"; done
