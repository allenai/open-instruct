#!/bin/bash
# Wait for a JobSet in namespace olmo to finish; save every pod's log under $RUNS_DIR/<run>/ (default ~/tpu-runs).
#   watch_jobset.sh <jobset> <run-name> [max-minutes]
export PATH=~/.local/bin:$PATH
C=gke_ai2-olmo_us-central2-b_gke-tpu-v4; JS=$1; RUN=$2; MAX=${3:-120}
OUT=${RUNS_DIR:-~/tpu-runs}/$RUN; mkdir -p "$OUT"
save_logs() {
  for p in $(kubectl --context $C -n olmo get pods -l jobset.sigs.k8s.io/jobset-name=$JS -o name 2>/dev/null); do
    kubectl --context $C -n olmo logs "$p" -c maxtext > "$OUT/$(basename $p).log" 2>&1
  done
}
for i in $(seq 1 "$MAX"); do
  st=$(kubectl --context $C -n olmo get jobset $JS -o jsonpath='{.status.terminalState}' 2>/dev/null)
  ph=$(kubectl --context $C -n olmo get pods -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null | awk '{print $3}' | sort | uniq -c | tr '\n' ' ')
  echo "$(TZ=America/Los_Angeles date +%H:%M) state=${st:-running} pods: $ph"
  [ -n "$st" ] && break
  # Save logs as soon as pods are running, so a deleted JobSet does not lose them.
  (( i % 5 == 0 )) && save_logs
  sleep 60
done
save_logs
echo "final state: ${st:-timeout}"; ls -la "$OUT"
