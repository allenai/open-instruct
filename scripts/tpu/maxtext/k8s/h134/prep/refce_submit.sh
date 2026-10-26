#!/bin/bash
# H134 P2: when the 14T HF export lands (convert.log EXPORT-OK), submit the HF step-1 CE reference job once; then wait for
# both refce_c11-s{0,1}.json and log their CE. Ids and losses only.
W=/weka/oe-eval-default/abhishekr/watch/tpu/h134; R=$W/refce; L=$W/convert/convert.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $R/refce.log >> $W/events.log; }
until grep -q 'EXPORT-OK' $L; do grep -qE 'EXPORT-FAIL|VENV-FAIL' $L && { log "export failed; no refce submitted"; exit 1; }; sleep 60; done
[ -f $R/experiment.id ] || { id=$(beaker experiment create --workspace ai2/olmo-instruct $R/beaker-refce.yaml --format json 2>>$R/submit.err | python3 -c "import json,sys;d=json.load(sys.stdin);print((d[0] if isinstance(d,list) else d)['id'])")
  [ -n "$id" ] || { log "refce submit failed (see submit.err)"; exit 2; }; echo $id > $R/experiment.id
  log "refce submitted: Beaker $id (ai2/olmo-instruct, jupiter, 1 GPU, high, minRuntime 5m, timeout 4h)"; }
until [ -f $R/refce_c11-s0.json ] && [ -f $R/refce_c11-s1.json ]; do sleep 120; done
for s in 0 1; do log "REFCE c11-s$s: $(python3 -c "import json;d=json.load(open('$R/refce_c11-s$s.json'));print('ce', round(d['ce'],6), 'loss_tokens', d['loss_tokens'])")"; done
