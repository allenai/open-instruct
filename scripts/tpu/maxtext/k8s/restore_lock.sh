#!/bin/bash
# Restore lock for TPU launches (see /weka/oe-eval-default/abhishekr/watch/tpu/RESTORE_LOCK.md). Source it, then:
#   restore_lock_apply <holder> <kustomize dir> [big]   # the default for a new launch: see below
#   restore_lock_activate <holder> <jobset> [big]       # the same for an existing, inactive JobSet
#   restore_lock_release <holder>                       # at step 1 or a terminal state, as before
#   restore_lock_end_big                                # after a 256-host restore ends: stamps big_restore_end
# Lower level: restore_lock_acquire <holder> blocks until the lock is free or its holder is stale (> 45 min); it also
# refreshes the holder's timestamp when called by the holder.
# State: $RL_STATE holds "free" or "<holder> <epoch seconds>"; flock on $RL_MUTEX makes check-and-set atomic. Nothing is
# deleted: release writes "free".
# This file is installed at /weka/oe-eval-default/abhishekr/watch/tpu/restore_lock.sh; the copy in open-instruct
# (scripts/tpu/maxtext/k8s/restore_lock.sh on x16-tpu-overlays) is the source.
RL_DIR=${RL_DIR:-/weka/oe-eval-default/abhishekr/watch/tpu}
RL_STATE=${RL_STATE:-$RL_DIR/restore.lock}; RL_MUTEX=${RL_MUTEX:-$RL_DIR/restore.lock.mutex}; RL_LOG=${RL_LOG:-$RL_DIR/restore.lock.log}
RL_STALE=${RL_STALE:-2700}
RL_K=${RL_K:-kubectl --context gke_ai2-olmo_us-central2-b_gke-tpu-v4 -n olmo}
RL_ADMIT_WAIT=${RL_ADMIT_WAIT:-600}  # seconds an active workload may wait for admission while holding the lock
RL_RETRY=${RL_RETRY:-300}            # seconds between admission attempts while inactive
restore_lock_acquire() {
  local me=$1 now h t
  while :; do
    exec 8>>"$RL_MUTEX"; flock 8
    now=$(date +%s); read -r h t < "$RL_STATE" 2>/dev/null
    if [ -z "$h" ] || [ "$h" = free ] || [ "$h" = "$me" ] || [ $((now - ${t:-0})) -gt $RL_STALE ]; then
      [ -n "$h" ] && [ "$h" != free ] && [ "$h" != "$me" ] && echo "$(date -u +%FT%TZ) restore lock: $h stale ($((now - t)) s), taken by $me" >> "$RL_LOG"
      echo "$me $now" > "$RL_STATE"; flock -u 8; exec 8>&-
      [ "$h" = "$me" ] || echo "$(date -u +%FT%TZ) restore lock: acquired by $me" >> "$RL_LOG"; return 0
    fi
    flock -u 8; exec 8>&-; sleep 30
  done
}
restore_lock_release() {
  local me=$1 h t
  exec 8>>"$RL_MUTEX"; flock 8
  read -r h t < "$RL_STATE" 2>/dev/null
  if [ "$h" = "$me" ]; then echo free > "$RL_STATE"; echo "$(date -u +%FT%TZ) restore lock: released by $me" >> "$RL_LOG"; fi
  flock -u 8; exec 8>&-
}

# Big-restore spacing (RESTORE_LOCK.md, 10-23 04:20 PT): no 256-host restore before big_restore_quiet_until, nor within
# 30 min of the last one's end.
restore_lock_wait_big() {
  until [ "$(date +%s)" -ge "$(cat "$RL_DIR/big_restore_quiet_until" 2>/dev/null || echo 0)" ] &&
        [ "$(date +%s)" -ge $(( $(cat "$RL_DIR/big_restore_end" 2>/dev/null || echo 0) + 1800 )) ]; do sleep 60; done
}
restore_lock_end_big() { date +%s > "$RL_DIR/big_restore_end"; }

_rl_wl() { $RL_K get workloads -o name 2>/dev/null | grep "/jobset-$1-[0-9a-f]*$" | head -1; }
_rl_admitted() { [ "$($RL_K get "$1" -o jsonpath='{.status.conditions[?(@.type=="Admitted")].status}' 2>/dev/null)" = True ]; }
_rl_active() { $RL_K patch "$1" --type=merge -p "{\"spec\":{\"active\":$2}}" > /dev/null; }

# Lock before apply (2026-10-23, after gc32's bs-32K run was admitted while another job held the lock): a workload is
# active only while <holder> holds the lock, so its restore cannot start without it. Each attempt takes the lock, makes
# the workload active, and waits up to RL_ADMIT_WAIT s for admission. On admission it returns 0 holding the lock; the
# caller releases it at step 1 or a terminal state, as before. Otherwise it makes the workload inactive (an admission
# that lands during that patch is evicted while the lock is still held), releases the lock and retries after RL_RETRY s.
# With "big", every attempt first waits for the big-restore spacing.
restore_lock_activate() {
  local me=$1 js=$2 big=${3:-} wl t0
  wl=$(_rl_wl "$js"); [ -n "$wl" ] || { echo "restore_lock_activate: no workload for $js" >&2; return 1; }
  while :; do
    [ "$big" = big ] && restore_lock_wait_big
    restore_lock_acquire "$me"
    _rl_active "$wl" true; t0=$(date +%s)
    until _rl_admitted "$wl" || [ $(( $(date +%s) - t0 )) -ge "$RL_ADMIT_WAIT" ]; do sleep 15; restore_lock_acquire "$me"; done
    _rl_admitted "$wl" && { echo "$(date -u +%FT%TZ) restore lock: $js admitted under $me" >> "$RL_LOG"; return 0; }
    _rl_active "$wl" false; restore_lock_release "$me"
    echo "$(date -u +%FT%TZ) restore lock: $js not admitted in ${RL_ADMIT_WAIT}s; inactive, retry in ${RL_RETRY}s" >> "$RL_LOG"
    sleep "$RL_RETRY"
  done
}
# Apply happens under the lock, so an instant admission is covered too; the workload is then handled as above.
restore_lock_apply() {
  local me=$1 dir=$2 big=${3:-} js i
  js=$($RL_K kustomize "$dir" 2>/dev/null | awk '/^kind: JobSet/{f=1} f&&/^  name:/{print $2; exit}')
  [ -n "$js" ] || { echo "restore_lock_apply: no JobSet in $dir" >&2; return 1; }
  [ "$big" = big ] && restore_lock_wait_big
  restore_lock_acquire "$me"
  $RL_K apply -k "$dir" > /dev/null || { restore_lock_release "$me"; return 1; }
  echo "$(date -u +%FT%TZ) restore lock: $js applied under $me" >> "$RL_LOG"
  for i in $(seq 1 40); do [ -n "$(_rl_wl "$js")" ] && break; sleep 3; done
  restore_lock_activate "$me" "$js" "$big"
}
