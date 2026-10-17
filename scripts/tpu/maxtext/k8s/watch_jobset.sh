#!/bin/bash
# Wait for a JobSet in namespace olmo to finish; save every pod's log under $RUNS_DIR/<run>/ (default ~/tpu-runs).
#   watch_jobset.sh <jobset> <run-name> [max-minutes]
set -euo pipefail
C=${KUBE_CONTEXT:-gke_ai2-olmo_us-central2-b_gke-tpu-v4}
JS=${1:?jobset}; RUN=${2:?run-name}; MAX=${3:-120}
[[ "$MAX" =~ ^[1-9][0-9]*$ ]] || { echo "max-minutes must be a positive integer" >&2; exit 2; }
OUT=${RUNS_DIR:-$HOME/tpu-runs}/$RUN; mkdir -p "$OUT"
kubectl_args=(--context "$C" -n olmo)
get_with_retry() {
  local attempt status response
  for attempt in 1 2 3; do
    if response=$(kubectl "${kubectl_args[@]}" --request-timeout=30s "$@"); then
      printf '%s\n' "$response"
      return 0
    else
      status=$?
    fi
    echo "Kubernetes read failed (attempt $attempt/3)" >&2
    if (( attempt < 3 )); then sleep 5; fi
  done
  return "$status"
}
save_logs() {
  local pods p result=0
  pods=$(get_with_retry get pods -l "jobset.sigs.k8s.io/jobset-name=$JS" -o name) || return 1
  for p in $pods; do
    if kubectl "${kubectl_args[@]}" logs "$p" -c maxtext > "$OUT/$(basename "$p").log.tmp"; then
      mv "$OUT/$(basename "$p").log.tmp" "$OUT/$(basename "$p").log"
    else
      echo "Could not save logs for $p; keeping the previous snapshot" >&2
      result=1
    fi
  done
  return "$result"
}
st=""
for i in $(seq 1 "$MAX"); do
  # Retry transient reads, then save what logs remain before reporting an API/authentication failure.
  st=$(get_with_retry get jobset "$JS" -o jsonpath='{.status.terminalState}') || {
    status=$?
    save_logs || echo "Final log snapshot failed" >&2
    exit "$status"
  }
  pods=$(get_with_retry get pods -l "jobset.sigs.k8s.io/jobset-name=$JS" --no-headers) || {
    status=$?
    save_logs || echo "Final log snapshot failed" >&2
    exit "$status"
  }
  ph=$(printf '%s\n' "$pods" | awk '{print $3}' | sort | uniq -c | tr '\n' ' ')
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') state=${st:-running} pods: $ph"
  [ -n "$st" ] && break
  # Save logs as soon as pods are running, so a deleted JobSet does not lose them.
  if (( i % 5 == 0 )); then
    save_logs || echo "Log snapshot failed" >&2
  fi
  sleep 60
done
log_status=0
save_logs || { echo "Final log snapshot failed" >&2; log_status=1; }
echo "final state: ${st:-timeout}"; ls -la "$OUT"
case "$st" in
  Completed) exit "$log_status" ;;
  "") exit 124 ;;
  *) exit 1 ;;
esac
