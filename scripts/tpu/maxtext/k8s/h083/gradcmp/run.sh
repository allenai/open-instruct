#!/bin/bash
# H083 1-step gradient comparison (ledger-manager ruling 2026-10-23): runs gradcmp-old and gradcmp-bs (one 4x4x8 each, queued
# after X1 leg A), each taking the restore lock from admission to step 1, then reads the three weight sets with compare.py
# under the lock (~140 GB from GCS) and posts GRAD-CMP to the x16prod event log. Run from a copy of this directory on Hammond:
#   setsid -f nohup ./run.sh > run.out 2>&1 < /dev/null
set -u
export CLOUDSDK_CONFIG=/tmp/gcloud-cfg CLOUDSDK_CORE_ACCOUNT=tpu-posttrain-agent@ai2-olmo.iam.gserviceaccount.com
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcloud-cfg/tpu-posttrain-agent.json
K="kubectl --context gke_ai2-olmo_us-central2-b_gke-tpu-v4 -n olmo"
D=$(cd "$(dirname "$0")" && pwd); EV=/weka/oe-eval-default/abhishekr/watch/tpu/x16prod/events.log
P0=gs://ai2-olmo-dev/post-training/checkpoints/olmo35-small-4t-emo-step5961/0/items
SCR=gs://ai2-olmo-dev/post-training/scratch
source /weka/oe-eval-default/abhishekr/watch/tpu/restore_lock.sh
ev() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D/run.log" >> "$EV"; }
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/run.log"; }

state() { # Completed | Failed | running
  $K get jobset "$1" -o jsonpath='{range .status.conditions[?(@.status=="True")]}{.type}{" "}{end}' 2>/dev/null |
    grep -oE 'Completed|Failed' | head -1 || true
}
pod0() { $K get pods -l jobset.sigs.k8s.io/jobset-name="$1" -o name 2>/dev/null | grep -- '-slice-0-0-' | head -1; }
grab() { # all pods' logs (BATCH-HASH, LOSS-PARTS, steps, errors) into logs/<js>/
  mkdir -p "$D/logs/$1"
  for p in $($K get pods -l jobset.sigs.k8s.io/jobset-name="$1" -o name 2>/dev/null); do
    $K logs "$p" -c maxtext 2>/dev/null > "$D/logs/$1/${p#pod/}.log"
  done
}

run_one() {
  local js=$1 w s i
  $K apply -k "$D/overlays/$js" >> "$D/run.log" 2>&1 || { ev "GRAD-CMP $js apply failed"; return 1; }
  ev "GRAD-CMP $js applied (queued for a 4x4x8)"
  until w=$($K get workloads -o name 2>/dev/null | grep -- "/jobset-$js-") && [ -n "$w" ]; do sleep 30; done
  until [ "$($K get "$w" -o jsonpath='{.status.conditions[?(@.type=="Admitted")].status}' 2>/dev/null)" = True ]; do sleep 60; done
  ev "GRAD-CMP $js admitted"; restore_lock_acquire "$js" >> "$D/run.log" 2>&1
  for i in $(seq 1 180); do
    s=$(state "$js"); [ -n "$s" ] && break
    p=$(pod0 "$js"); [ -n "$p" ] && $K logs "$p" -c maxtext --tail=200 2>/dev/null | grep -q "completed step: 1," && break
    sleep 30
  done
  restore_lock_release "$js" >> "$D/run.log" 2>&1; log "$js lock released"
  for i in $(seq 1 120); do s=$(state "$js"); [ -n "$s" ] && break; sleep 30; done
  grab "$js"; ev "GRAD-CMP $js ${s:-no terminal state after 60 min}"
  [ "$s" = Completed ]
}

$K get workloads -o name | grep -q '/jobset-h089-x1-mt-s1-448-' || { ev "GRAD-CMP not started: X1 leg A h089-x1-mt-s1-448 not applied yet"; exit 1; }
run_one h083-gradcmp-old & a=$!
run_one h083-gradcmp-bs & b=$!
wait $a; ra=$?; wait $b; rb=$?
[ $ra -eq 0 ] && [ $rb -eq 0 ] || { ev "GRAD-CMP stopped: a run did not complete (old rc=$ra, bs rc=$rb); see $D/logs"; exit 1; }

# Same batch and the same forward pass in both runs, from the captured logs.
python3 - "$D" > "$D/checks.txt" <<'PY'
import glob, re, sys
d = sys.argv[1]
def get(js, pat):
    out = set()
    for f in glob.glob(f"{d}/logs/{js}/*.log"):
        out |= set(re.findall(pat, open(f, errors="replace").read()))
    return out
for name, pat in [("hash", r"BATCH-HASH n=1 host=\d+ sha256=\w+"), ("loss", r"LOSS-PARTS step=1 [^\n]*")]:
    a, b = get("h083-gradcmp-old", pat), get("h083-gradcmp-bs", pat)
    print(f"{name}: old {len(a)} bs {len(b)} equal {a == b and len(a) > 0}")
PY
log "$(cat "$D/checks.txt" | tr '\n' ';')"

items() { # the orbax item dir under checkpoints/1 that holds the decoder weights
  for m in $(gcloud storage ls "$SCR/$1/checkpoints/1/**/_METADATA" 2>/dev/null); do
    gcloud storage cat "$m" 2>/dev/null | grep -q decoder && { dirname "$m"; return; }
  done
}
IO=$(items h083-gradcmp-old); IB=$(items h083-gradcmp-bs)
[ -n "$IO" ] && [ -n "$IB" ] || { ev "GRAD-CMP stopped: step-1 checkpoint not found (old '$IO', bs '$IB')"; exit 1; }
log "items: old $IO, bs $IB"
restore_lock_acquire gradcmp-compare >> "$D/run.log" 2>&1
/tmp/gcvenv/bin/python "$D/compare.py" "$P0" "$IO" "$IB" > "$D/result.json" 2> "$D/compare.err"; rc=$?
restore_lock_release gradcmp-compare >> "$D/run.log" 2>&1
[ $rc -eq 0 ] || { ev "GRAD-CMP stopped: compare.py failed (rc $rc), see $D/compare.err"; exit 1; }
ev "$(python3 - "$D/result.json" "$D/checks.txt" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); w = r["worst"][0]
checks = " ".join(l.strip().replace(" ", "_") for l in open(sys.argv[2]))
print(f"GRAD-CMP h083 old vs bs3ec936c3: rel={r['rel']:.3e} (floor {r['floor']:.1e}) worst={w['tensor']} {w['rel']:.3e} "
      f"(floor {w['floor']:.1e}) bit-identical tensors {r['bit_identical_tensors']}/{r['tensors']} -> {'PASS' if r['pass'] else 'FAIL'} (<= 1e-5); {checks}")
PY
)"
