#!/bin/bash
# Stream process 0's pod (the one printing REQUIRED_RBF) of a running JobSet into logs/<js>/<pod>.follow.log; kubectl logs -f
# follows kubelet log rotation, so no drop record is lost between xw2 captures. Re-finds process 0 after every restart or
# re-admission; exits when the JobSet is terminal. H123's xw/follow.sh with H134's directory.   ./follow.sh <js>
export CLOUDSDK_CORE_ACCOUNT=tpu-posttrain-agent@ai2-olmo.iam.gserviceaccount.com
K="kubectl --context gke_ai2-olmo_us-central2-b_gke-tpu-v4 -n olmo"; JS=$1; cd /weka/oe-eval-default/abhishekr/watch/tpu/h134/xw; mkdir -p logs/$JS
KEEP="BATCH-HASH|LOSS-PARTS|REQUIRED_RBF|completed step|schedule, LR|Restored|Resume|Traceback|Error|RESOURCE_EXHAUSTED"
while :; do
  p=""; for q in $($K get pods -l jobset.sigs.k8s.io/jobset-name=$JS -o name 2>/dev/null); do $K logs $q --tail=300 2>/dev/null | grep -q REQUIRED_RBF && { p=${q#pod/}; break; }; done
  [ -z "$p" ] && { [ -n "$($K get jobset $JS -o jsonpath='{.status.terminalState}' 2>/dev/null)" ] && exit 0; sleep 60; continue; }
  echo "$(date -u +%FT%TZ) $JS following $p"
  $K logs -f --since=3m pod/$p 2>/dev/null | grep --line-buffered -E "$KEEP" >> logs/$JS/$p.follow.log
  echo "$(date -u +%FT%TZ) $JS follow of $p ended"
  [ -n "$($K get jobset $JS -o jsonpath='{.status.terminalState}' 2>/dev/null)" ] && { echo "$(date -u +%FT%TZ) $JS terminal, follower done"; exit 0; }
  sleep 10
done
