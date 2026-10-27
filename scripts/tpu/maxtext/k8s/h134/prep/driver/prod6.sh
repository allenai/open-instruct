#!/bin/bash
# H134 mm-ckpt-sft driver for all six arms (tpu-worker, 2026-10-26; ledger-manager's order after #747): B1 s0, B1 s1, B2 s0,
# B2 s1, B0 s0, B0 s1 (arms.conf, in that order), each on one 4x4x8 with the shared ep8 x fsdp16 mesh. One persistent event
# loop (60 s); state in prod/state/ so a restarted driver resumes.
#  - An arm is considered only once arms/<arm>-s<seed>.ready exists: its base, HF step-1 reference and overlays are in place
#    and recorded (for B0: the rounded-tensor sha256 in its receipt). The flag is written by hand, not by this driver.
#  - Per arm: preflight pf-<arm>-c11-s<s> (20 steps; gate_pf.py: P2 step-1 CE vs HF, P3 factor and drops, P4 hashes 1-20)
#    must PASS before its production apply. Production applies follow arms.conf's order among ready arms: an arm waits while
#    an earlier ready arm that has not failed is still unapplied. Preflights go in the same order, at most one start per pass.
#  - Every apply/activation asserts the overlay's rendered hash (arms.conf) and its init path; O is explicit.
#  - Nothing starts before H131's c11-s2 and c11-s3 JobSets exist. Every restore is under restore.lock, taken only when free
#    (never stolen) and, for a new apply or a re-activation, only when no one has acquired it in the last 30 min (checked
#    under the lock's mutex). A JobSet not admitted within 10 min is deactivated (confirmed) and retried later.
#  - A restart or a Kueue re-admission (running pods 0 -> >0) stamps the lock if free, else the JobSet is deactivated and
#    re-activated later under the lock; the lock is held until the run logs a step past the one it restored from.
#    Limitation: polling (60 s) means such a restore starts up to ~60 s before its stamp lands; an eviction and
#    re-admission within one poll would go unseen.
#  - GoogleEgressBandwidth in a JobSet's logs since its last (re)start stops it. Stops count once the workload reads inactive.
#  - P5: xw/follow.sh and xw/xw2.py per production JobSet from its apply until terminal; receipt_p5.py at the end.
# The P4 x16 reference (pfd-*) is a diagnostic (ledger-manager 2026-10-26) and is not in this launch path.
set -u
export CLOUDSDK_CONFIG=/tmp/gcloud-cfg CLOUDSDK_CORE_ACCOUNT=tpu-posttrain-agent@ai2-olmo.iam.gserviceaccount.com
source /weka/oe-eval-default/abhishekr/watch/tpu/restore_lock.sh
W=/weka/oe-eval-default/abhishekr/watch/tpu/h134; cd $W; mkdir -p prod pf xw prod/state arms
ST=$W/prod/state; CONF=$W/arms.conf
O=$W/oi-h134/scripts/tpu/maxtext/k8s/h134          # open-instruct h134-b0-overlays, checked out at $W/oi-h134
GIT=$W/oi-h134
PY=/tmp/h083mt/bin/python
K="kubectl --context gke_ai2-olmo_us-central2-b_gke-tpu-v4 -n olmo"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $W/prod/prod.log >> $W/events.log; }
get() { cat $ST/$1 2>/dev/null; }
put() { echo "$2" > $ST/$1.tmp && mv $ST/$1.tmp $ST/$1; }
setactive() { local w; w=$($K get workloads -o name 2>/dev/null | grep "/jobset-$1-[0-9a-f]*$" | head -1); [ -n "$w" ] && $K patch $w --type merge -p "{\"spec\":{\"active\":$2}}" > /dev/null 2>&1; }
astate() { # active | inactive | unknown (a failed lookup is never read as inactive)
  local w a; w=$($K get workloads -o name 2>/dev/null) || { echo unknown; return; }; w=$(echo "$w" | grep "/jobset-$1-[0-9a-f]*$" | head -1)
  [ -n "$w" ] || { echo unknown; return; }; a=$($K get $w -o jsonpath='{.spec.active}' 2>/dev/null) || { echo unknown; return; }
  [ "$a" = false ] && echo inactive || echo active; }
deact() { setactive $1 false; sleep 5; [ "$(astate $1)" = inactive ]; }
admitted() { local w; w=$($K get workloads -o name 2>/dev/null | grep "/jobset-$1-[0-9a-f]*$" | head -1); [ -n "$w" ] && [ "$($K get $w -o jsonpath='{.status.conditions[?(@.type=="Admitted")].status}' 2>/dev/null)" = True ]; }
terminal() { $K get jobset $1 -o jsonpath='{.status.terminalState}' 2>/dev/null; }
running() { $K get pods -l jobset.sigs.k8s.io/jobset-name=$1 --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l; }
dir_of() { case $1 in h134-pf-*) echo pf/logs/$1;; *) echo prod/$1;; esac; }
maxstep() { grep -ohE "completed step: [0-9]+" $(dir_of $1)/*.log 2>/dev/null | grep -oE "[0-9]+$" | sort -n | tail -1; }
assert_render() { # $1 overlay dir, $2 expected hash, $3 init
  local h; h=$($K kustomize $O/$1 2>/dev/null | sha256sum | cut -c1-16)
  [ "$h" = "$2" ] || { log "ASSERT: $1 renders $h, expected $2; not applying"; return 1; }
  $K kustomize $O/$1 | grep -qF "LOAD_PARAMETERS_PATH: $3" || { log "ASSERT: $1 lacks init $3"; return 1; }
  $K kustomize $O/$1 | grep -qF "cloud.google.com/gke-tpu-topology: 4x4x8" || { log "ASSERT: $1 is not a 4x4x8 overlay"; return 1; }; }
lock_take() { # $1 holder, $2 spaced|now: take or refresh if free or ours; "spaced" also needs 30 min since anyone's last acquisition
  local h t last ok=1; exec 8>>"$RL_MUTEX"; flock 8; read -r h t < "$RL_STATE" 2>/dev/null
  if [ "$h" = "$1" ]; then echo "$1 $(date +%s)" > "$RL_STATE"; flock -u 8; exec 8>&-; return 0; fi
  if [ -z "$h" ] || [ "$h" = free ]; then
    if [ "$2" = spaced ]; then
      last=$(grep -E 'restore lock: (acquired|stamped) by' "$RL_LOG" | tail -1 | cut -c1-20); last=$(date -d "$last" +%s 2>/dev/null || echo 0)
      [ $(( $(date +%s) - last )) -ge 1800 ] || ok=0
    fi
    if [ $ok = 1 ]; then echo "$1 $(date +%s)" > "$RL_STATE"; echo "$(date -u +%FT%TZ) restore lock: acquired by $1 ($2)" >> "$RL_LOG"; flock -u 8; exec 8>&-; return 0; fi
  fi
  flock -u 8; exec 8>&-; return 1; }
lock_holder() { read -r h t < "$RL_STATE" 2>/dev/null; echo "${h:-free}"; }
save() { # $1 jobset: append kept lines of every pod since the last save (+60 s overlap; readers dedupe)
  local d=$(dir_of $1) now=$(date +%s) last since="" ok=1 pods; mkdir -p $d; last=$(cat $d/.last 2>/dev/null || echo 0)
  [ "$last" -gt 0 ] && since="--since=$(( now - last + 60 ))s"
  pods=$($K get pods -l jobset.sigs.k8s.io/jobset-name=$1 -o name 2>/dev/null) || ok=0
  for p in $pods; do
    $K logs $p $since > $d/.fetch 2>/dev/null || { ok=0; continue; }
    grep -aE "BATCH-HASH|LOSS-PARTS|REQUIRED_RBF|completed step|schedule, LR|Restored|GoogleEgressBandwidth|Traceback|Error|nan|inf" $d/.fetch >> $d/$(basename $p).log
  done
  [ $ok = 1 ] && echo $now > $d/.last; }   # on any fetch failure the cursor stays, so the next save re-reads the gap
egress() { [ -f $ST/$1.t ] && find $(dir_of $1) -newer $ST/$1.t -name '*.log' -print0 2>/dev/null | xargs -0 grep -l GoogleEgressBandwidth 2>/dev/null | head -1; }
stop_js() { log "STOP $1: $2"; setactive $1 false; put $1.stopping "$2"; }
ensure_watchers() { # $1 production jobset
  local p; p=$(get follow_$1.pid); { [ -n "$p" ] && kill -0 $p 2>/dev/null; } || { (cd xw && setsid -f nohup bash ./follow.sh $1 >> follow_$1.out 2>&1 < /dev/null)
    sleep 1; p=$(ps -eo pid,args | awk -v j=$1 '$2=="bash" && $3=="./follow.sh" && $4==j {print $1}' | tail -1); put follow_$1.pid "$p"; log "follower for $1 pid $p"; }
  p=$(get xw_$1.pid); { [ -n "$p" ] && kill -0 $p 2>/dev/null; } || { (cd xw && setsid -f nohup python3 xw2.py 16304 $1 >> xw_$1.out 2>&1 < /dev/null)
    sleep 1; p=$(ps -eo pid,args | awk -v j=$1 '$2=="python3" && $3=="xw2.py" && $5==j {print $1}' | tail -1); put xw_$1.pid "$p"; log "xw2.py drop watcher for $1 pid $p"; }; }
start_js() { # $1 overlay, $2 jobset, $3 output prefix, $4 expected hash, $5 init, $6 held-from step: apply or activate under the lock
  assert_render $1 $4 $5 || { put $2.phase blocked; return 1; }
  if ! $K get jobset $2 >/dev/null 2>&1; then
    local n; n=$(gcloud storage ls "$3/" 2>/dev/null | wc -l); [ "$n" = 0 ] || { log "$2 output prefix $3 not empty ($n); not applying"; put $2.phase blocked; return 1; }
    lock_take $2 spaced || return 1
    $K apply -k $O/$1 > prod/apply_$2.log 2>&1 || { log "$2 apply failed"; restore_lock_release $2; return 1; }
    log "$2 applied (oi-h134 $(git -C $GIT rev-parse --short HEAD), rendered $($K kustomize $O/$1 | sha256sum | cut -c1-16)) under the lock"
  else
    lock_take $2 spaced || return 1
    setactive $2 true; log "$2 activated under the lock"
  fi
  touch $ST/$2.t; put $2.hash $4; put $2.held "${6:-0}"; put $2.deadline $(( $(date +%s) + 600 )); put $2.rs "$($K get jobset $2 -o jsonpath='{.status.restarts}' 2>/dev/null || echo 0)"
  put $2.run 0; put $2.phase running; return 0; }
watch_js() { # $1 jobset: one pass over a started JobSet
  local J=$1 st r nr held m why
  if [ -f $ST/$J.stopping ]; then [ "$(astate $J)" = inactive ] || { setactive $J false; return; }
    log "$J inactive after stop ($(get $J.stopping))"; put $J.phase stopped; rm -f $ST/$J.stopping $ST/$J.held; [ "$(lock_holder)" = "$J" ] && restore_lock_release $J; return; fi
  if [ -f $ST/$J.stopping_for_lock ]; then deact $J && { rm -f $ST/$J.held $ST/$J.stopping_for_lock; put $J.phase inactive; log "$J inactive (waiting for the lock)"; }; return; fi
  save $J; st=$(terminal $J)
  if [ -n "$st" ]; then [ -f $ST/$J.held ] && { restore_lock_release $J; rm -f $ST/$J.held; }; rm -f $ST/$J.deadline; put $J.phase "terminal:$st"; log "$J terminal: $st"; return; fi
  if [ ! -f $ST/$J.deadline ] && [ "$(astate $J)" = inactive ]; then log "$J was deactivated outside this driver (xw2.py's #550 stop or by hand): stopped"; put $J.phase stopped; [ "$(lock_holder)" = "$J" ] && restore_lock_release $J; rm -f $ST/$J.held; return; fi
  if [ -f $ST/$J.deadline ]; then
    if admitted $J; then rm -f $ST/$J.deadline
    elif [ $(date +%s) -gt $(get $J.deadline) ]; then
      deact $J || { log "$J not admitted in 10 min; deactivation not yet confirmed, retrying (lock kept)"; return; }
      restore_lock_release $J; rm -f $ST/$J.deadline $ST/$J.held; put $J.phase inactive; log "$J not admitted in 10 min: inactive, lock released, retry later"; return; fi
  fi
  [ -n "$(egress $J)" ] && { stop_js $J "GoogleEgressBandwidth in its logs since its last (re)start"; return; }
  r=$($K get jobset $J -o jsonpath='{.status.restarts}' 2>/dev/null); r=${r:-0}; nr=$(running $J)
  why=""; [ "$r" -gt "$(get $J.rs || echo 0)" ] && why="RESTART $r"
  [ "$(get $J.run || echo 0)" = 0 ] && [ "$nr" -gt 0 ] && [ -z "$why" ] && [ ! -f $ST/$J.held ] && why="RE-ADMISSION (pods 0 -> $nr)"
  put $J.rs $r; put $J.run $nr
  if [ -n "$why" ]; then
    m=$(maxstep $J); touch $ST/$J.t
    if lock_take $J now; then put $J.held "${m:-0}"; log "$why of $J after step ${m:-0}: lock held until it trains past it"
    else log "$why of $J while $(lock_holder) holds the lock: deactivating, re-activate under the lock later"; put $J.stopping_for_lock 1
      deact $J && { rm -f $ST/$J.held $ST/$J.stopping_for_lock; put $J.phase inactive; }; return; fi
  fi
  if [ -f $ST/$J.held ]; then held=$(get $J.held); m=$(maxstep $J)
    if [ "${m:-0}" -gt "$held" ]; then restore_lock_release $J; rm -f $ST/$J.held; log "$J trained past step $held (step $m): lock released"
    else lock_take $J now > /dev/null || log "WARN: $J holds no lock during its restore (holder $(lock_holder))"; fi
  fi; }
[ -f $CONF ] || { log "no $CONF; not starting"; exit 1; }
log "driver start (pid $$), arms: $(grep -vE '^\s*(#|$)' $CONF | awk '{printf "%s-s%s ", $1, $2}'), oi-h134 $(git -C $GIT rev-parse --short HEAD)"
while :; do
  h131=0; $K get jobset h131-olmo3-7b-c11-s2 >/dev/null 2>&1 && $K get jobset h131-olmo3-7b-c11-s3 >/dev/null 2>&1 && h131=1
  busy=0; started=0; ahead=0   # ahead: an earlier ready, not-failed arm is still waiting for its production apply
  mapfile -t LINES < <(grep -vE '^\s*(#|$)' $CONF)   # read up front: nothing below may consume the config as stdin
  for line in "${LINES[@]}"; do
    read -r A S PFH PRH INIT REF <<< "$line"; PJ=h134-pf-$A-c11-s$S; J=h134-$A-c11-s$S
    if [ ! -f arms/$A-s$S.ready ]; then busy=1; continue; fi
    [ -f "$REF" ] || { [ -f $ST/$A-s$S.noref ] || log "$A-s$S ready but its HF reference $REF is missing"; touch $ST/$A-s$S.noref; busy=1; ahead=1; continue; }
    # state is bound to the render it was made with: a changed overlay (e.g. B0's fp32 -> bf16 init) never reuses old state
    for x in "$PJ $PFH" "$J $PRH"; do set -- $x; if [ -f $ST/$1.hash ] && [ "$(get $1.hash)" != "$2" ]; then
      [ "$(get $1.phase)" = blocked ] || { log "$1: state was made with render $(get $1.hash), arms.conf says $2; blocked (clear prod/state/$1.* by hand)"; put $1.phase blocked; }; fi; done
    case "$(get $PJ.phase)" in
      ""|inactive) busy=1; [ $h131 = 1 ] && [ $started = 0 ] && start_js pf-$A-c11-s$S $PJ gs://ai2-olmo-dev/post-training/scratch/olmo35-sft-$PJ $PFH $INIT "$(maxstep $PJ)" && started=1;;
      running) busy=1; watch_js $PJ;;
      terminal:*) if [ ! -f $ST/$PJ.gate ]; then
          if [ "$(get $PJ.phase)" = terminal:Completed ] && grep -qh "completed step: 20," pf/logs/$PJ/*.log 2>/dev/null; then $PY $W/gate_pf.py $A $S $REF; rc=$?
          else rc=1; log "$PJ ended $(get $PJ.phase) without step 20: preflight FAIL"; fi
          [ $rc = 2 ] && { log "$PJ gate incomplete after Completed: treated as FAIL"; rc=1; }
          put $PJ.gate $rc; log "PREFLIGHT $A-s$S: $([ $rc = 0 ] && echo PASS || echo FAIL) (pf/gate_pf_${A}_s$S.json)"; fi;;
      stopped|blocked) :;;
    esac
    g=$(get $PJ.gate)
    if [ -z "$g" ]; then case "$(get $PJ.phase)" in stopped|blocked) :;; *) ahead=1;; esac; continue; fi
    [ "$g" = 0 ] || continue
    case "$(get $J.phase)" in
      ""|inactive) busy=1
        if [ $ahead = 0 ] && [ $h131 = 1 ] && [ $started = 0 ]; then
          start_js $A-c11-s$S $J gs://ai2-olmo-dev/post-training/runs/olmo35-sft-$J $PRH $INIT "$(maxstep $J)" && { started=1; ensure_watchers $J; log "PRODUCTION $A-s$S applied"; }
        fi; [ "$(get $J.phase)" = running ] || ahead=1;;
      running) busy=1; ensure_watchers $J; watch_js $J
        [ "$(get $J.phase)" = running ] && { out=$($PY pcheck.py $A $S $REF); rc=$?; echo "$(date -u +%FT%TZ) $out" >> prod/status.log
          [ $rc = 11 ] && stop_js $J "INCOMPLETE: $out"; [ $rc = 10 ] && stop_js $J "UNSTABLE: $out"; };;
      terminal:*) [ -f $ST/$J.receipt ] || { log "$J: $($PY pcheck.py $A $S $REF); P5 $(python3 receipt_p5.py $J 16304)"; touch $ST/$J.receipt; };;
      blocked) ahead=1;;   # a passing arm that cannot apply keeps later arms waiting (needs a human)
      stopped) :;;
    esac
  done
  [ $busy = 0 ] && { log "no arm left to start or watch; driver done"; exit 0; }
  sleep 60
done
