#!/bin/bash
# H134 mm-ckpt-sft b0 driver (tpu-worker, 2026-10-26): arm b0-c11 (14T base olmo35-small-14t600b-mtlc-emo, H122's C11
# shards, H123's A0' recipe, 16,304 steps), seeds 0 and 1 on one 4x4x8 each. Pattern: H122/H123's prod.sh, plus:
#  - every apply asserts the overlay's rendered hash; O and the overlay dirs are explicit (no defaults);
#  - order: H103's lock and the H131 c11-s2/s3 applies first (ledger-manager); every apply under restore.lock
#    (RL_STALE=86400: wait for "free", never steal), at least 30 min after the previous acquisition by anyone;
#  - per seed, the P2/P3/P4 preflight pf-b0-c11-s<s> (20 steps) must PASS gate_pf.py before the production apply;
#  - P5: xw/follow.sh (JAX process 0, across restarts) and xw/xw2.py (#550 budget, 10-min-or-faster saver) from the
#    production apply on; receipt_p5.py at the end;
#  - the lock is taken (stamped) on JobSet restarts AND when a JobSet's running pods go from 0 to >0 (Kueue
#    re-admission), held until the run trains past its last step, refreshed while held;
#  - any GoogleEgressBandwidth in a pod log after an apply stops that JobSet (no restart loop).
# The P4 x16 reference (pfd-*) is a diagnostic and is not in this launch path.
set -u
export CLOUDSDK_CONFIG=/tmp/gcloud-cfg CLOUDSDK_CORE_ACCOUNT=tpu-posttrain-agent@ai2-olmo.iam.gserviceaccount.com
export RL_STALE=86400
source /weka/oe-eval-default/abhishekr/watch/tpu/restore_lock.sh
restore_lock_try() { # take or refresh the lock now if free or ours; else print the holder and return 1 (a restart's restore never waits)
  local h t; exec 8>>"$RL_MUTEX"; flock 8; read -r h t < "$RL_STATE" 2>/dev/null
  if [ -z "$h" ] || [ "$h" = free ] || [ "$h" = "$1" ]; then
    echo "$1 $(date +%s)" > "$RL_STATE"; flock -u 8; exec 8>&-; [ "$h" = "$1" ] || echo "$(date -u +%FT%TZ) restore lock: stamped by $1 (restart)" >> "$RL_LOG"; return 0; fi
  flock -u 8; exec 8>&-; echo "$h"; return 1; }
W=/weka/oe-eval-default/abhishekr/watch/tpu/h134; cd $W; mkdir -p prod pf xw
O=$W/oi-h134/scripts/tpu/maxtext/k8s/h134          # open-instruct h134-b0-overlays (checked out at $W/oi-h134)
GIT=$W/oi-h134; SEEDS=${SEEDS:-0 1}
declare -A EXPECT=( [pf-b0-c11-s0]=6553a179f59a2215 [pf-b0-c11-s1]=3eac04560c9b1dbf [b0-c11-s0]=928236fabc2555f2 [b0-c11-s1]=6a68c2fee209d77f )
INIT=gs://ai2-olmo-dev/post-training/checkpoints/olmo35-small-14t600b-mtlc-emo/h134-lc-step5961/0/items
PY=/tmp/h083mt/bin/python
K="kubectl --context gke_ai2-olmo_us-central2-b_gke-tpu-v4 -n olmo"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $W/prod/prod.log; echo "$(date -u +%FT%TZ) $*" >> $W/events.log; }
stop() { $K patch $($K get workloads -o name | grep "/jobset-$1-[0-9a-f]*$") --type merge -p '{"spec":{"active":false}}' 2>&1 | tail -1 | tee -a $W/prod/prod.log; }
assert_render() { local h; h=$($K kustomize $O/$1 2>/dev/null | sha256sum | cut -c1-16)
  [ "$h" = "${EXPECT[$1]}" ] || { log "ASSERT: $1 renders $h, expected ${EXPECT[$1]}; not applying"; return 1; }
  $K kustomize $O/$1 | grep -qF "LOAD_PARAMETERS_PATH: $INIT" || { log "ASSERT: $1 lacks the 14T init path"; return 1; }; }
save() { # $1 jobset, $2 dir: append kept lines of every pod since the last save (+5 min overlap)
  local now=$(date +%s) last=$(cat $2/.last 2>/dev/null || echo 0) since=""; mkdir -p $2
  [ "$last" -gt 0 ] && since="--since=$(( now - last + 300 ))s"
  for p in $($K get pods -l jobset.sigs.k8s.io/jobset-name=$1 -o name 2>/dev/null); do
    $K logs $p $since 2>/dev/null | grep -aE "BATCH-HASH|LOSS-PARTS|REQUIRED_RBF|completed step|schedule, LR|Restored|GoogleEgressBandwidth|Traceback|Error|nan|inf" >> $2/$(basename $p).log
  done; echo $now > $2/.last; }
egress() { grep -lsh GoogleEgressBandwidth $1/*.log 2>/dev/null | head -1; }
running() { $K get pods -l jobset.sigs.k8s.io/jobset-name=$1 --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l; }
wait_turn() { # $1 holder: H131's two controls applied, lock free, and >= 30 min since anyone's last acquisition
  until $K get jobset h131-olmo3-7b-c11-s2 >/dev/null 2>&1 && $K get jobset h131-olmo3-7b-c11-s3 >/dev/null 2>&1; do sleep 120; done
  while :; do
    local h t last; read -r h t < $RL_STATE
    last=$(grep -E 'restore lock: (acquired|stamped) by' $RL_LOG | tail -1 | cut -c1-20)
    last=$(date -d "$last" +%s 2>/dev/null || echo 0)
    [ "$h" = free ] && [ $(( $(date +%s) - last )) -ge 1800 ] && return 0
    sleep 60
  done; }
apply_one() { # $1 overlay dir name, $2 jobset, $3 output prefix: apply under the lock, hold it until step 1 (refreshing)
  assert_render $1 || return 1
  local n; n=$(gcloud storage ls "$3/" 2>/dev/null | wc -l); [ "$n" = 0 ] || { log "$2 output prefix $3 not empty ($n), not applying"; return 2; }
  wait_turn $2; assert_render $1 || return 1
  date +%s > $W/.apply_t_$2
  restore_lock_apply $2 $O/$1 > $W/prod/apply_$2.log 2>&1 || { log "$2 apply failed"; restore_lock_release $2; return 3; }
  log "$2 applied (oi-h134 $(git -C $GIT rev-parse --short HEAD), rendered $($K kustomize $O/$1 | sha256sum | cut -c1-16)), lock held until step 1"; }
wait_step1() { # $1 jobset, $2 log dir: refresh the lock while waiting; release at step 1 or a terminal state
  local t0=$(date +%s)
  while :; do
    save $1 $2; restore_lock_acquire $1
    grep -qh "LOSS-PARTS step=1 " $2/*.log 2>/dev/null && break
    [ -n "$(egress $2)" ] && { log "EGRESS: GoogleEgressBandwidth in $1's logs -> stopping it"; stop $1; restore_lock_release $1; return 4; }
    [ -n "$($K get jobset $1 -o jsonpath='{.status.terminalState}' 2>/dev/null)" ] && break
    [ $(( $(date +%s) - t0 )) -gt 5400 ] && { log "$1 no step 1 after 90 min -> stopping it"; stop $1; restore_lock_release $1; return 5; }
    sleep 60
  done
  save $1 $2; restore_lock_release $1; log "$1 lock released at step 1 ($(grep -h 'LOSS-PARTS step=1 ' $2/*.log | head -1 | grep -oE 'ce=[0-9.]+'))"; }
# ---- preconditions (P1: B0 is the registry entry; P2's reference must exist before any apply) ----
for s in $SEEDS; do [ -f $W/refce/refce_c11-s$s.json ] || { log "no HF step-1 reference refce_c11-s$s.json; not starting"; exit 1; }; done
gcloud storage ls $INIT/commit_success.txt >/dev/null 2>&1 || { log "init $INIT not committed on GCS; not starting"; exit 1; }
# ---- per seed: preflight (P2/P3/P4) then production ----
for s in $SEEDS; do
  PJ=h134-pf-b0-c11-s$s; J=h134-b0-c11-s$s
  if [ -f pf/gate_pf_s$s.json ] && grep -q '"pass": true' pf/gate_pf_s$s.json; then log "$PJ already PASS"
  else
    if ! $K get jobset $PJ >/dev/null 2>&1; then apply_one pf-b0-c11-s$s $PJ gs://ai2-olmo-dev/post-training/scratch/olmo35-sft-h134-pf-b0-c11-s$s || exit $?; fi
    wait_step1 $PJ pf/logs/$PJ || exit $?
    until [ -n "$($K get jobset $PJ -o jsonpath='{.status.terminalState}' 2>/dev/null)" ]; do save $PJ pf/logs/$PJ; sleep 60; done
    save $PJ pf/logs/$PJ; log "$PJ terminal: $($K get jobset $PJ -o jsonpath='{.status.terminalState}')"
    $PY $W/gate_pf.py $s; rc=$?
    [ $rc = 0 ] || { log "$PJ gate rc=$rc: not launching $J (see pf/gate_pf_s$s.json)"; continue; }
  fi
  if $K get jobset $J >/dev/null 2>&1; then log "$J exists, not reapplying"
  else
    apply_one b0-c11-s$s $J gs://ai2-olmo-dev/post-training/runs/olmo35-sft-h134-b0-c11-s$s || exit $?
    (cd xw && setsid -f nohup ./follow.sh $J >> follow_s$s.out 2>&1 < /dev/null)
    wait_step1 $J prod/s$s || { log "$J did not reach step 1 cleanly"; continue; }
  fi
done
# ---- one xw2.py drop watcher for the production JobSets that exist ----
JS=""; for s in $SEEDS; do $K get jobset h134-b0-c11-s$s >/dev/null 2>&1 && JS="$JS h134-b0-c11-s$s"; done
[ -z "$JS" ] && { log "no production JobSet; driver done"; exit 0; }
[ -f xw/xw.pid ] && ps -p $(cat xw/xw.pid) >/dev/null 2>&1 || { (cd xw && setsid -f nohup python3 xw2.py 16304 $JS >> xw.out 2>&1 < /dev/null); sleep 2
  ps -eo pid,args | awk '$2=="python3" && $3=="xw2.py"{print $1}' | tail -1 > xw/xw.pid; log "xw2.py drop watcher pid $(cat xw/xw.pid) on$JS"; }
# ---- monitor loop ----
declare -A RS=() HELD=() RUN=()
for s in $SEEDS; do J=h134-b0-c11-s$s; r=$($K get jobset $J -o jsonpath='{.status.restarts}' 2>/dev/null); RS[$s]=${r:-0}; RUN[$s]=$(running $J); done
while :; do
  live=0
  for s in $SEEDS; do J=h134-b0-c11-s$s
    $K get jobset $J >/dev/null 2>&1 || continue
    [ -f prod/s$s/TERMINAL ] && continue
    st=$($K get jobset $J -o jsonpath='{.status.terminalState}' 2>/dev/null)
    save $J prod/s$s
    r=$($K get jobset $J -o jsonpath='{.status.restarts}' 2>/dev/null); r=${r:-0}; nr=$(running $J)
    why=""; [ "$r" -gt "${RS[$s]}" ] && why="RESTART $r"; [ "${RUN[$s]}" = 0 ] && [ "$nr" -gt 0 ] && [ -z "$why" ] && why="RE-ADMISSION (pods 0 -> $nr)"
    RS[$s]=$r; RUN[$s]=$nr
    if [ -n "$why" ]; then
      date +%s > $W/.apply_t_$J
      if other=$(restore_lock_try $J); then HELD[$s]=$($PY pcheck.py $s --maxstep); log "$why of $J after step ${HELD[$s]}: restore lock held until it trains past it"
      else HELD[$s]=""; log "$why of $J; OVERLAP: $other holds the restore lock during this restore"; fi
    fi
    [ -n "${HELD[$s]:-}" ] && restore_lock_try $J >/dev/null   # refresh while held
    out=$($PY pcheck.py $s); rc=$?; echo "$(date -u +%FT%TZ) $out" >> prod/status.log
    if [ -n "${HELD[$s]:-}" ] && [ "$($PY pcheck.py $s --maxstep)" -gt "${HELD[$s]}" ]; then restore_lock_release $J; log "$J resumed past ${HELD[$s]}, lock released"; HELD[$s]=""; fi
    if [ -f $W/.apply_t_$J ] && [ -n "$(find prod/s$s -newer $W/.apply_t_$J -name '*.log' -print0 | xargs -0 grep -l GoogleEgressBandwidth 2>/dev/null | head -1)" ]; then
      log "EGRESS: GoogleEgressBandwidth in $J's logs since its last (re)start -> stopping it"; stop $J; touch prod/s$s/TERMINAL
      [ -n "${HELD[$s]:-}" ] && restore_lock_release $J; continue; fi
    [ $rc = 11 ] && { log "$out -> INCOMPLETE, stopping $J"; stop $J; touch prod/s$s/TERMINAL; [ -n "${HELD[$s]:-}" ] && restore_lock_release $J; continue; }
    [ $rc = 10 ] && { log "$out -> UNSTABLE, stopping $J"; stop $J; touch prod/s$s/TERMINAL; [ -n "${HELD[$s]:-}" ] && restore_lock_release $J; continue; }
    if [ -n "$st" ]; then save $J prod/s$s; log "$J terminal: $st; $($PY pcheck.py $s); P5 $(python3 receipt_p5.py $J 16304)"; touch prod/s$s/TERMINAL
      [ -n "${HELD[$s]:-}" ] && restore_lock_release $J; continue; fi
    live=1
  done
  [ $live = 0 ] && { log "all b0 seeds terminal; driver done"; exit 0; }
  sleep 300
done
