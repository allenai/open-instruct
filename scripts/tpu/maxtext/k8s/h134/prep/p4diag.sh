#!/bin/bash
# H134 P4 x16 diagnostic driver (tpu-worker, 2026-10-26; ledger-manager: "run H134's P4 x16 references on [a free 8x8x16]
# now, then release it ... under the restore lock, behind H134's production applies, and pair each with its 4x4x8 candidate
# whenever a 4x4x8 has a gap. They never hold a launch; a missing one is reported as missing."). Not in prod6.sh's path.
#  - x16 references pfd-x16-<arm>-s<s> (8x8x16, 6 steps), one at a time, in order b1-s0 b1-s1 b2-s0 b2-s1 b0-s0 b0-s1; each
#    ep8 candidate pfd-ep8-<arm>-s<s> (4x4x8, the production mesh) only after its reference is done, when a 4x4x8 is free.
#  - Yields to prod6.sh: never takes the lock while an H134 arm has a passed preflight with its production unapplied, a
#    preflight running or waiting to start, or a production restore holding the lock (prod6's state, read only). With
#    pfdiag/relax_capacity present, a preflight that cannot start for want of a free 4x4x8 does not hold back an x16
#    reference (it does hold back candidates, which need the same slices).
#  - Lock: restore.lock, taken only when free (never stolen), refreshed every pass while held; x16 starts also wait for the
#    big-restore spacing (big_restore_quiet_until, big_restore_end + 30 min). Apply or activate under the lock; not admitted
#    in 10 min: deactivated, lock released, retried later. Released at step 1 (x16: big_restore_end stamped). A Kueue
#    re-admission (running pods 0 -> >0) stamps the lock if free, else the JobSet is deactivated.
#  - Before admission a run yields to prod6 (deactivated, lock released) whenever prod6 gets work; an admitted restore is
#    never interrupted, but one with no step 1 within 75 min of admission, or no step 6 within 2 h, is stopped. Three admission timeouts: missing.
#  - Each run is stopped (workload spec.active=false, confirmed before the lock is released) once LOSS-PARTS and
#    REQUIRED_RBF for step 6 are saved, or ends on its own. Pod logs are saved to pfdiag/logs/<js>/ every pass (kept lines
#    only); a failed attempt's logs move to <js>.att<n>. An eviction, a re-admission, a Failed JobSet or
#    GoogleEgressBandwidth counts as a failed attempt: one retry, then the pair is reported missing. Any x16 attempt that
#    was admitted stamps big_restore_end when its restore ends (step 1) or the attempt stops.
#  - Limitation: every acquisition here restarts prod6's 30-min "spaced" window; only starting when prod6 has nothing to
#    do bounds the delay to ~30 min after a diagnostic apply.
#  - When both runs of a pair are done: gate_p4diag2.py <arm> <seed>. Events go to h134/events.log tagged P4DIAG.
set -u
export CLOUDSDK_CONFIG=/tmp/gcloud-cfg CLOUDSDK_CORE_ACCOUNT=tpu-posttrain-agent@ai2-olmo.iam.gserviceaccount.com
source /weka/oe-eval-default/abhishekr/watch/tpu/restore_lock.sh
W=/weka/oe-eval-default/abhishekr/watch/tpu/h134; cd $W
D=$W/pfdiag; ST=$D/state; PST=$W/prod/state; CONF=$W/arms.conf; mkdir -p $ST $D/logs
O=$W/oi-h134/scripts/tpu/maxtext/k8s/h134
K="kubectl --context gke_ai2-olmo_us-central2-b_gke-tpu-v4 -n olmo"
ORDER="b1-s0 b1-s1 b2-s0 b2-s1 b0-s0 b0-s1"
KEEP="BATCH-HASH|LOSS-PARTS|REQUIRED_RBF|completed step|schedule, LR|Restored|GoogleEgressBandwidth|GCSFS-RETRY|GCSFS-STAGGER|routed_bias|Traceback|Error"
log() { echo "$(date -u +%FT%TZ) P4DIAG $*" | tee -a $D/p4diag.log >> $W/events.log; }
get() { cat $ST/$1 2>/dev/null; }
put() { echo "$2" > $ST/$1.tmp && mv $ST/$1.tmp $ST/$1; }
pget() { cat $PST/$1 2>/dev/null; }
wl() { $K get workloads -o name 2>/dev/null | grep "/jobset-$1-[0-9a-f]*$" | head -1; }
setactive() { local w; w=$(wl $1); [ -n "$w" ] && $K patch $w --type merge -p "{\"spec\":{\"active\":$2}}" > /dev/null 2>&1; }
astate() { local w a; w=$($K get workloads -o name 2>/dev/null) || { echo unknown; return; }; w=$(echo "$w" | grep "/jobset-$1-[0-9a-f]*$" | head -1)
  [ -n "$w" ] || { echo none; return; }; a=$($K get $w -o jsonpath='{.spec.active}' 2>/dev/null) || { echo unknown; return; }
  [ "$a" = false ] && echo inactive || echo active; }
deact() { setactive $1 false; sleep 5; [ "$(astate $1)" = inactive ]; }
admitted() { local w; w=$(wl $1); [ -n "$w" ] && [ "$($K get $w -o jsonpath='{.status.conditions[?(@.type=="Admitted")].status}' 2>/dev/null)" = True ]; }
terminal() { $K get jobset $1 -o jsonpath='{.status.terminalState}' 2>/dev/null; }
nrun() { $K get pods -l jobset.sigs.k8s.io/jobset-name=$1 --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l; }
free_slices() { # $1 topology: node pools of that topology minus admitted workloads on it (any queue); prints -1 on a failed read
  local pools adm
  pools=$($K get nodes -L cloud.google.com/gke-tpu-topology,cloud.google.com/gke-nodepool --no-headers 2>/dev/null | awk -v t=$1 '$(NF-1)==t {print $NF}' | sort -u | wc -l)
  adm=$($K get workloads -A -o json 2>/dev/null | python3 -c 'import json,sys
n=0
for w in json.load(sys.stdin)["items"]:
  c={x["type"]:x["status"] for x in w.get("status",{}).get("conditions",[])}
  if c.get("Finished")!="True" and c.get("Admitted")=="True" and sys.argv[1] in json.dumps(w["status"].get("admission",{})): n+=1
print(n)' $1 2>/dev/null) || { echo -1; return; }
  [ "$pools" -gt 0 ] && [ -n "$adm" ] && echo $(( pools - adm )) || echo -1; }
holder() { local h t; read -r h t < "$RL_STATE" 2>/dev/null; echo "${h:-free}"; }
lk_take() { # $1 holder, $2 note: take if free or ours (refresh); never steal
  local h t r=1; exec 8>>"$RL_MUTEX"; flock 8; read -r h t < "$RL_STATE" 2>/dev/null
  if [ "$h" = "$1" ]; then echo "$1 $(date +%s)" > "$RL_STATE"; r=0
  elif [ -z "$h" ] || [ "$h" = free ]; then echo "$1 $(date +%s)" > "$RL_STATE"; echo "$(date -u +%FT%TZ) restore lock: $2 by $1" >> "$RL_LOG"; r=0; fi
  flock -u 8; exec 8>&-; return $r; }
big_ok() { local now; now=$(date +%s)
  [ $now -ge "$(cat $RL_DIR/big_restore_quiet_until 2>/dev/null || echo 0)" ] && [ $now -ge $(( $(cat $RL_DIR/big_restore_end 2>/dev/null || echo 0) + 1800 )) ]; }
init_of() { awk -v a=$1 -v s=$2 '$1==a && $2==s {print $5}' $CONF; }
prod6_waiting() { # prints the reason and returns 0 while prod6 has something to do first; $1 = x16|ep8
  local A S r PJ J pp g jp free4=""
  while read -r A S r; do
    [ -f arms/$A-s$S.ready ] || continue
    PJ=h134-pf-$A-c11-s$S; J=h134-$A-c11-s$S; pp=$(pget $PJ.phase); g=$(pget $PJ.gate); jp=$(pget $J.phase)
    case "$pp" in blocked|stopped) continue;; esac
    if [ -z "$g" ]; then
      case "$pp" in
        ""|inactive) if [ "$1" = x16 ] && [ -f $D/relax_capacity ]; then [ -n "$free4" ] || free4=$(free_slices 4x4x8); [ "$free4" = 0 ] && continue; fi
          echo "$A-s$S preflight waiting to start"; return 0;;
        *) echo "$A-s$S preflight $pp"; return 0;;
      esac
    fi
    [ "$g" = 0 ] || continue
    case "$jp" in ""|inactive) echo "$A-s$S passed its preflight, production unapplied"; return 0;; esac
    [ -f $PST/$J.held ] && { echo "$J restore holds the lock"; return 0; }
  done < <(grep -vE '^\s*(#|$)' $CONF)
  return 1; }
save() { # $1 jobset: each pod's full log, kept lines only, overwriting that pod's file (a failed fetch keeps the old one)
  local d=$D/logs/$1; mkdir -p $d
  $K get pods -l jobset.sigs.k8s.io/jobset-name=$1 -o name 2>/dev/null | sed 's|^pod/||' |
    xargs -r -P 16 -I{} bash -c "$K logs {} 2>/dev/null > $d/.{}.f && grep -aE '$KEEP' $d/.{}.f > $d/{}.log.tmp && mv $d/{}.log.tmp $d/{}.log; rm -f $d/.{}.f"; }
podfiles() { $K get pods -l jobset.sigs.k8s.io/jobset-name=$1 -o name 2>/dev/null | sed "s|^pod/|$D/logs/$1/|; s|\$|.log|"; }
has() { grep -qhE "$2" $D/logs/$1/*.log 2>/dev/null; }                  # this attempt's logs (earlier attempts are archived)
has_now() { local f; f=$(podfiles $1); [ -n "$f" ] && grep -qhE "$2" $f 2>/dev/null; }   # the pods that exist now
step6() { has $1 "LOSS-PARTS step=6 " && has $1 "REQUIRED_RBF step=6 "; }
step1_now() { has_now $1 "completed step: [1-9]"; }   # as prod6: trained past step 0
check_render() { # $1 overlay, $2 init, $3 topology: prints the rendered hash, nonzero if the init or topology is wrong
  local r; r=$($K kustomize $O/$1 2>/dev/null) || return 1
  echo "$r" | grep -qF "LOAD_PARAMETERS_PATH: $2" || return 1; echo "$r" | grep -qF "cloud.google.com/gke-tpu-topology: $3" || return 1
  echo "$r" | sha256sum | cut -c1-16; }
kind_of() { case $1 in *pfd-x16-*) echo x16;; *) echo ep8;; esac; }
archive() { # $1 jobset, $2 suffix: move this attempt's logs aside so a later attempt's checks never read them
  [ -d $D/logs/$1 ] && [ -n "$(ls -A $D/logs/$1 2>/dev/null)" ] && mv $D/logs/$1 $D/logs/$1.$2; mkdir -p $D/logs/$1; }
end_big() { [ -f $ST/$1.adm ] || return 0; rm -f $ST/$1.adm; [ "$(kind_of $1)" = x16 ] && restore_lock_end_big; return 0; }

start() { # $1 kind (x16|ep8), $2 arm, $3 seed: apply or activate under the lock
  local kind=$1 a=$2 s=$3 ov js topo h n
  ov=pfd-$kind-$a-s$s; js=h134-$ov; topo=$([ $kind = x16 ] && echo 8x8x16 || echo 4x4x8)
  h=$(check_render $ov "$(init_of $a $s)" $topo) || { log "$js: render check failed (init $(init_of $a $s), $topo); marked missing"; put $js.phase missing; return 1; }
  if ! $K get jobset $js >/dev/null 2>&1; then
    n=$(gcloud storage ls "gs://ai2-olmo-dev/post-training/scratch/olmo35-sft-$js/" 2>/dev/null | wc -l)
    [ "$n" = 0 ] || { log "$js: output prefix not empty ($n); marked missing"; put $js.phase missing; return 1; }
    lk_take $js acquired || return 1
    $K apply -k $O/$ov > $D/apply_$js.log 2>&1 || { log "$js apply failed"; restore_lock_release $js; return 1; }
    log "$js applied (oi-h134 $(git -C $W/oi-h134 rev-parse --short HEAD), rendered $h) under the lock"
  else
    [ -n "$(terminal $js)" ] && { log "$js is terminal ($(terminal $js)) and cannot rerun under its name; marked missing"; put $js.phase missing; return 1; }
    lk_take $js acquired || return 1
    setactive $js true; log "$js activated under the lock (attempt $(( $(get $js.att || echo 0) + 1 )))"
  fi
  mkdir -p $D/logs/$js; rm -f $ST/$js.pods $ST/$js.adm
  put $js.deadline $(( $(date +%s) + 600 )); put $js.held 1; put $js.phase admitting; return 0; }

stop() { # $1 jobset, $2 phase after (done|inactive|missing), $3 why: the lock is released only once the workload reads inactive
  put $1.after "$2"; put $1.why "$3"; put $1.phase stopping; stopping $1; }
stopping() { local js=$1 after a
  [ -f $ST/$js.held ] && lk_take $js acquired > /dev/null   # keep it fresh (RL_STALE) until the workload reads inactive
  if $K get jobset $js >/dev/null 2>&1 && [ -z "$(terminal $js)" ]; then a=$(astate $js)
    if [ "$a" != inactive ]; then admitted $js && touch $ST/$js.adm   # an admission racing the stop still ends a big restore
      [ "$a" = active ] && setactive $js false; sleep 5
      [ "$(astate $js)" = inactive ] || { log "$js: deactivation not yet confirmed (workload $(astate $js)); retrying (lock kept)"; return; }; fi; fi
  [ -f $ST/$js.held ] && { restore_lock_release $js; rm -f $ST/$js.held; }
  end_big $js; after=$(get $js.after); put $js.phase $after
  [ "$after" = inactive ] && put $js.next $(( $(date +%s) + 1200 ))
  log "$js: $(get $js.why); now $after$([ "$after" = inactive ] && echo ', retry in 20 min or later')"; }

fail() { # $1 jobset, $2 why: an attempt that counts; one retry, then missing
  local js=$1 at; at=$(( $(get $js.att || echo 0) + 1 )); put $js.att $at; archive $js att$at
  if [ $at -ge 2 ]; then stop $js missing "$2 (attempt $at)"; else stop $js inactive "$2 (attempt $at)"; fi; }

watch() { # $1 jobset: one pass
  local js=$1 st now pods t
  now=$(date +%s)
  case "$(get $js.phase)" in
    stopping) stopping $js;;
    admitting)
      if admitted $js; then touch $ST/$js.adm; put $js.rdl $(( now + 4500 )); put $js.rdl6 $(( now + 7200 )); put $js.phase running; log "$js admitted"
      elif w=$(prod6_waiting $(kind_of $js)); then stop $js inactive "yielding to prod6 before admission ($w)"
      elif [ $now -gt "$(get $js.deadline)" ]; then t=$(( $(get $js.tmo || echo 0) + 1 )); put $js.tmo $t
        if [ $t -ge 3 ]; then stop $js missing "not admitted in 10 min ($t times)"; else stop $js inactive "not admitted in 10 min ($t of 3)"; fi
      else lk_take $js acquired > /dev/null; fi;;
    running)
      save $js; st=$(terminal $js)
      if step6 $js; then save $js; stop $js done "steps 1-6 logged; stopped"; return; fi
      [ "$st" = Completed ] && { stop $js done "Completed (step 6 not in its saved logs: the gate reports it incomplete)"; return; }
      [ "$st" = Failed ] && { put $js.att 1; fail $js "Failed (maxRestarts 0: the same name cannot rerun)"; return; }
      grep -qh GoogleEgressBandwidth $D/logs/$js/*.log 2>/dev/null && { fail $js "GoogleEgressBandwidth in its logs"; return; }
      admitted $js || { fail $js "evicted before step 6"; return; }
      if pods=$($K get pods -l jobset.sigs.k8s.io/jobset-name=$js -o name 2>/dev/null | sort) && [ -n "$pods" ]; then
        # pods seen earlier that are gone while the JobSet is neither terminal nor evicted: an eviction and re-admission between passes
        touch $ST/$js.pods
        if [ -n "$(comm -23 $ST/$js.pods <(echo "$pods"))" ]; then fail $js "re-admitted between passes (earlier pods gone)"; return; fi
        echo "$pods" | sort -u - $ST/$js.pods > $ST/$js.pods.tmp && mv $ST/$js.pods.tmp $ST/$js.pods; fi
      [ -n "$(get $js.rdl)" ] || put $js.rdl $(( now + 4500 )); [ -n "$(get $js.rdl6)" ] || put $js.rdl6 $(( now + 7200 ))
      [ $now -gt "$(get $js.rdl6)" ] && { fail $js "no step 6 within 2 h of admission"; return; }
      if [ -f $ST/$js.held ]; then
        if step1_now $js; then restore_lock_release $js; rm -f $ST/$js.held; end_big $js; log "$js logged step 1: lock released"
        elif [ $now -gt "$(get $js.rdl)" ]; then fail $js "no step 1 within 75 min of admission"
        else lk_take $js acquired > /dev/null || log "WARN: $js holds no lock during its restore (holder $(holder))"; fi
      fi;;
  esac; }


log "driver start (pid $$), order $ORDER, oi-h134 $(git -C $W/oi-h134 rev-parse --short HEAD)"
while :; do
  active=""; for r in $ORDER; do for k in x16 ep8; do js=h134-pfd-$k-${r%-s*}-s${r#*-s}
    case "$(get $js.phase)" in admitting|running|stopping) watch $js; case "$(get $js.phase)" in admitting|running|stopping) active="$active $js";; esac;; esac; done; done
  # gates for finished pairs
  for r in $ORDER; do a=${r%-s*}; s=${r#*-s}; x=h134-pfd-x16-$a-s$s; e=h134-pfd-ep8-$a-s$s
    if [ "$(get $x.phase)" = done ] && [ "$(get $e.phase)" = done ] && [ -z "$(get $a-s$s.gate)" ]; then
      out=$(python3 $W/gate_p4diag2.py $a $s 2>&1); rc=$?; put $a-s$s.gate $rc; log "gate $a-s$s rc $rc: $out" > /dev/null; fi; done
  if [ -z "$active" ]; then
    nx=""; for r in $ORDER; do js=h134-pfd-x16-${r%-s*}-s${r#*-s}; case "$(get $js.phase)" in done|missing) ;; *) nx=$r; break;; esac; done
    if [ -n "$nx" ]; then
      why=$(prod6_waiting x16) && { [ "$(get wait)" = "$why" ] || { log "x16 $nx waits: prod6 first ($why)"; put wait "$why"; }; }
      if [ -z "$why" ] && [ $(date +%s) -ge "$(get h134-pfd-x16-$nx.next || echo 0)" ] && big_ok && [ "$(holder)" = free ] && [ "$(free_slices 8x8x16)" -gt 0 ]; then rm -f $ST/wait; start x16 ${nx%-s*} ${nx#*-s}; fi
    else
      [ -f $ST/x16_released ] || { left=""; for r in $ORDER; do js=h134-pfd-x16-${r%-s*}-s${r#*-s}; [ "$(astate $js)" = active ] && { setactive $js false; left="$left $js"; }; done
        [ -z "$left" ] && { touch $ST/x16_released; log "all six x16 references done or missing: $(for r in $ORDER; do echo -n "$r=$(get h134-pfd-x16-${r%-s*}-s${r#*-s}.phase) "; done); no pfd-x16 workload active, 8x8x16 released"; }; }
      ne=""; for r in $ORDER; do x=h134-pfd-x16-${r%-s*}-s${r#*-s}; e=h134-pfd-ep8-${r%-s*}-s${r#*-s}
        [ "$(get $x.phase)" = done ] || continue; case "$(get $e.phase)" in done|missing) ;; *) ne=$r; break;; esac; done
      if [ -n "$ne" ]; then
        why=$(prod6_waiting ep8) && { [ "$(get wait)" = "$why" ] || { log "ep8 $ne waits: prod6 first ($why)"; put wait "$why"; }; }
        if [ -z "$why" ] && [ $(date +%s) -ge "$(get h134-pfd-ep8-$ne.next || echo 0)" ] && [ "$(holder)" = free ] && [ "$(free_slices 4x4x8)" -gt 0 ]; then rm -f $ST/wait; start ep8 ${ne%-s*} ${ne#*-s}; fi
      else
        log "done: $(for r in $ORDER; do echo -n "$r x16=$(get h134-pfd-x16-${r%-s*}-s${r#*-s}.phase) ep8=$(get h134-pfd-ep8-${r%-s*}-s${r#*-s}.phase) gate=$(get $r.gate) "; done)"; exit 0
      fi
    fi
  fi
  sleep 60
done
