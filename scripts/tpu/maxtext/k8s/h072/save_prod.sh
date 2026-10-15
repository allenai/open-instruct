#!/bin/bash
# H072 production log saver: every 10 min, keeps each pod's BATCH-HASH / LOSS-PARTS / HBM / resume / error lines in
# logs/<jobset>/<pod>.log (pods are renamed on restart, so earlier attempts keep their own files) until all four end.
cd "$(dirname "$0")"; K="kubectl --context gke_ai2-olmo_us-central2-b_gke-tpu-v4 -n olmo"
JS="h072-e2-s0 h072-e2-s1 h072-e4-s0 h072-e4-s1"
while true; do
  left=0
  for js in $JS; do
    mkdir -p logs/$js
    for p in $($K get pods -l jobset.sigs.k8s.io/jobset-name=$js -o name 2>/dev/null); do
      $K logs $p 2>/dev/null | grep -E "BATCH-HASH|LOSS-PARTS|HBM |Restored params|Resume: skipped|STEP-GUARD|Traceback|Error|RESOURCE_EXHAUSTED" > logs/$js/${p#pod/}.tmp && mv logs/$js/${p#pod/}.tmp logs/$js/${p#pod/}.log
    done
    st=$($K get jobset $js -o jsonpath='{.status.terminalState}' 2>/dev/null); [ -z "$st" ] && left=$((left+1))
  done
  [ $left = 0 ] && { echo "all ended"; for js in $JS; do echo "$js $($K get jobset $js -o jsonpath='{.status.terminalState}')"; done; break; }
  sleep 600
done
