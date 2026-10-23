#!/bin/bash
# H083 fp32 identity test (ledger-manager ruling 2026-10-23): the 1-step gradient comparison of gradcmp/ with fp32
# activations and highest matmul precision (overlays ../gc32-{old,bs}-s<len>, made by ../mk_gc32.py). It tries sequence
# 65536 first and steps down to 32768, then 16384, if HBM runs out. Both images always use the same shape. It waits until
# both H104 X1 leg-A JobSets (h104-x1-mt-s{0,1}-*) are applied, or until GO exists here. Each run holds the restore lock
# from admission to step 1, and the compare (compare.py) holds it again. Posts GRAD-CMP-FP32 to the x16prod event log.
# Idempotent: rerun it after a Hammond restart and it picks up from the JobSets' state. All state lives on WEKA, in this dir.
#   cd /weka/oe-eval-default/abhishekr/watch/tpu/gc32 && setsid -f nohup ./run.sh >> run.out 2>&1 < /dev/null
set -u
export CLOUDSDK_CORE_ACCOUNT=tpu-posttrain-agent@ai2-olmo.iam.gserviceaccount.com
if [ -d /tmp/gcloud-cfg ]; then export CLOUDSDK_CONFIG=/tmp/gcloud-cfg; fi
for k in /tmp/gcloud-cfg/tpu-posttrain-agent.json /root/.config/gcloud/tpu-posttrain-agent.json; do
  [ -f "$k" ] && { export GOOGLE_APPLICATION_CREDENTIALS=$k; break; }
done
K="kubectl --context gke_ai2-olmo_us-central2-b_gke-tpu-v4 -n olmo"
D=$(cd "$(dirname "$0")" && pwd); EV=/weka/oe-eval-default/abhishekr/watch/tpu/x16prod/events.log
P0=gs://ai2-olmo-dev/post-training/checkpoints/olmo35-small-4t-emo-step5961/0/items
SCR=gs://ai2-olmo-dev/post-training/scratch
BF16=/weka/oe-eval-default/abhishekr/watch/tpu/gradcmp/result.json
source /weka/oe-eval-default/abhishekr/watch/tpu/restore_lock.sh
ev() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D/run.log" >> "$EV"; }
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/run.log"; }
[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] || { ev "GRAD-CMP-FP32 stopped: no tpu-posttrain-agent key found"; exit 1; }
$K auth whoami 2>&1 | grep -q tpu-posttrain-agent || { ev "GRAD-CMP-FP32 stopped: kubectl auth failed"; exit 1; }
if [ ! -x /tmp/gcvenv/bin/python ]; then
  UV_CACHE_DIR=/tmp/gc-uvcache uv venv -q /tmp/gcvenv -p 3.11 && VIRTUAL_ENV=/tmp/gcvenv UV_CACHE_DIR=/tmp/gc-uvcache uv pip install -q tensorstore numpy
fi

exists() { $K get jobset "$1" -o name > /dev/null 2>&1; }
state() { $K get jobset "$1" -o jsonpath='{range .status.conditions[?(@.status=="True")]}{.type}{" "}{end}' 2>/dev/null | grep -oE 'Completed|Failed' | head -1 || true; }
pod0() { $K get pods -l jobset.sigs.k8s.io/jobset-name="$1" -o name 2>/dev/null | grep -- '-slice-0-0-' | head -1; }
grab() {
  mkdir -p "$D/logs/$1"
  for p in $($K get pods -l jobset.sigs.k8s.io/jobset-name="$1" -o name 2>/dev/null); do
    $K logs "$p" -c maxtext 2>/dev/null > "$D/logs/$1/${p#pod/}.log.tmp" && mv "$D/logs/$1/${p#pod/}.log.tmp" "$D/logs/$1/${p#pod/}.log"
  done
}
# HBM only: a VMEM overflow is a compile limit (attempt 1 hit "Used 17.09M of 16.00M vmem" at every length) and a smaller
# shape doesn't fix it, so it stops the ladder as a failure.
oom() { grep -qlE "Ran out of memory in memory space hbm|exceeds the HBM|Attempting to (allocate|reserve).*hbm" "$D"/logs/"$1"/*.log 2>/dev/null; }

gate() { # behind both H104 X1 leg-A drivers
  until [ -f "$D/GO" ] || { $K get workloads -o name 2>/dev/null | grep -q '/jobset-h104-x1-mt-s1-' &&
                            $K get workloads -o name 2>/dev/null | grep -q '/jobset-h104-x1-mt-s0-'; }; do sleep 120; done
}

run_one() { # -> Completed | OOM | Failed
  local js=$1 w s i
  if [ -f "$D/state/$js" ]; then cat "$D/state/$js"; return; fi
  if ! exists "$js"; then
    gate
    $K apply -k "$D/overlays/$js" >> "$D/run.log" 2>&1 || { echo Failed; return; }
    ev "GRAD-CMP-FP32 $js applied (queued for a 4x4x8)"
  fi
  until w=$($K get workloads -o name 2>/dev/null | grep -- "/jobset-$js-") && [ -n "$w" ]; do sleep 30; done
  until [ -n "$(state "$js")" ] || [ "$($K get "$w" -o jsonpath='{.status.conditions[?(@.type=="Admitted")].status}' 2>/dev/null)" = True ]; do sleep 60; done
  if [ -z "$(state "$js")" ]; then
    log "$js admitted"; restore_lock_acquire "$js" >> "$D/run.log" 2>&1
    for i in $(seq 1 240); do
      s=$(state "$js"); [ -n "$s" ] && break
      p=$(pod0 "$js"); [ -n "$p" ] && $K logs "$p" -c maxtext --tail=200 2>/dev/null | grep -q "completed step: 1," && break
      sleep 30
    done
    restore_lock_release "$js" >> "$D/run.log" 2>&1
  fi
  for i in $(seq 1 120); do s=$(state "$js"); [ -n "$s" ] && break; sleep 30; done
  grab "$js"
  if [ "$s" = Completed ]; then r=Completed; elif oom "$js"; then r=OOM; else r=Failed; fi
  mkdir -p "$D/state"; echo $r > "$D/state/$js"; ev "GRAD-CMP-FP32 $js $r"; echo $r
}

shape=""
# Shapes in order. Attempt 1 (plain, 16 MiB scoped VMEM) and attempt 2 (-v24) stopped on VMEM; attempt 3 (-v24t) also halves
# the ragged-dot tiles.
for S in ${GC32_SHAPES:-s64k-v24t s32k-v24t s16k-v24t}; do
  ro=$(run_one h083-gc32-old-$S); [ "$ro" = OOM ] && continue
  [ "$ro" = Completed ] || { ev "GRAD-CMP-FP32 stopped: h083-gc32-old-$S failed (not OOM); see $D/logs"; exit 1; }
  rb=$(run_one h083-gc32-bs-$S); [ "$rb" = OOM ] && { ev "GRAD-CMP-FP32 bs OOMed at $S where old fit; stepping both down"; continue; }
  [ "$rb" = Completed ] || { ev "GRAD-CMP-FP32 stopped: h083-gc32-bs-$S failed (not OOM); see $D/logs"; exit 1; }
  shape=$S; break
done
[ -n "$shape" ] || { ev "GRAD-CMP-FP32 stopped: out of HBM at every shape tried (${GC32_SHAPES:-s64k-v24t s32k-v24t s16k-v24t})"; exit 1; }

python3 - "$D" "$shape" > "$D/checks.txt" <<'PY'
import glob, re, sys
d, s = sys.argv[1:3]
def get(js, pat):
    out = set()
    for f in glob.glob(f"{d}/logs/{js}/*.log"):
        out |= set(re.findall(pat, open(f, errors="replace").read()))
    return out
for name, pat in [("hash", r"BATCH-HASH n=1 host=\d+ sha256=\w+"), ("loss", r"LOSS-PARTS step=1 [^\n]*")]:
    a, b = get(f"h083-gc32-old-{s}", pat), get(f"h083-gc32-bs-{s}", pat)
    print(f"{name}: old {len(a)} bs {len(b)} equal {a == b and len(a) > 0}")
PY
log "$(tr '\n' ';' < "$D/checks.txt")"
items() { for m in $(gcloud storage ls "$SCR/$1/checkpoints/1/**/_METADATA" 2>/dev/null); do gcloud storage cat "$m" 2>/dev/null | grep -q decoder && { dirname "$m"; return; }; done; }
IO=$(items h083-gc32-old-$shape); IB=$(items h083-gc32-bs-$shape)
[ -n "$IO" ] && [ -n "$IB" ] || { ev "GRAD-CMP-FP32 stopped: step-1 checkpoint not found (old '$IO', bs '$IB')"; exit 1; }
if [ ! -s "$D/result.json" ]; then
  restore_lock_acquire gc32-compare >> "$D/run.log" 2>&1
  /tmp/gcvenv/bin/python "$D/compare.py" "$P0" "$IO" "$IB" > "$D/result.json.tmp" 2> "$D/compare.err"; rc=$?
  restore_lock_release gc32-compare >> "$D/run.log" 2>&1
  [ $rc -eq 0 ] || { ev "GRAD-CMP-FP32 stopped: compare.py failed (rc $rc), see $D/compare.err"; exit 1; }
  mv "$D/result.json.tmp" "$D/result.json"
fi
python3 - "$D/result.json" "$BF16" "$shape" "$D/checks.txt" "$D/RECEIPT.md" > "$D/event.txt" <<'PY'
import json, sys
f, b, shape, checks, out = sys.argv[1:6]
r, q = json.load(open(f)), json.load(open(b))
seq = {"s64k": 65536, "s32k": 32768, "s16k": 16384}[shape.split("-")[0]]
vmem = {"v24": "24 MiB", "v24t": "24 MiB, ragged-dot fwd tiles 256/512/512"}.get(shape.split("-")[-1], "16 MiB")
qb = {w["tensor"]: w["rel"] for w in q["worst"]}
fp = r["per_tensor_rel"]
lines = ["# H083 fp32 identity test: receipt", "",
         f"Shape: 16 sequences x {seq} tokens, olmo35-ep-as-cp fsdp16 x ep8 on one 4x4x8, one plain-SGD step (LR 1e8, no clip); "
         f"scoped VMEM {vmem} (bf16 test: 16 MiB, the same in both images either way).",
         "Images by digest: old sha256:7d7f4938..., bs3ec936c3 sha256:c858123f... (KDA sharding pins, maxtext c5f1251d).",
         f"Checks: {open(checks).read().strip().replace(chr(10), '; ')}", "",
         "| | bf16 (activations bf16, default precision, megablox) | fp32 (activations fp32, highest, ragged_dot) |",
         "|---|---|---|",
         f"| whole model rel | {q['rel']:.3e} | {r['rel']:.3e} |",
         f"| read-out floor | {q['floor']:.1e} | {r['floor']:.1e} |",
         f"| bit-identical tensors | {q['bit_identical_tensors']}/{q['tensors']} | {r['bit_identical_tensors']}/{r['tensors']} |", "",
         "Worst tensors (rel = |g_bs - g_old| / |g_old|):", "", "| tensor | bf16 | fp32 |", "|---|---|---|"]
names = list(dict.fromkeys([w["tensor"] for w in q["worst"][:5]] + [w["tensor"] for w in r["worst"][:5]]))
for n in names:
    lines.append(f"| {n} | {qb[n]:.3e} |" if n in qb else f"| {n} | (not in bf16 top 10) |")
    lines[-1] += f" {fp[n]:.3e} |"
verdict = "PASS: class B, the pins are value identities" if r["pass"] else "FAIL: math change"
lines += ["", f"Rule (ledger-manager, fixed before the result): rel <= 1e-5 in fp32 -> class B; else math change. **{verdict}.**"]
open(out, "w").write("\n".join(lines) + "\n")
w = r["worst"][0]
print(f"GRAD-CMP-FP32 h083 old vs bs3ec936c3 at seq {seq}, vmem {vmem}: rel={r['rel']:.3e} (floor {r['floor']:.1e}; bf16 was {q['rel']:.3e}) "
      f"worst={w['tensor']} {w['rel']:.3e} bit-identical {r['bit_identical_tensors']}/{r['tensors']} -> "
      f"{'PASS' if r['pass'] else 'FAIL'} (<= 1e-5); receipt gc32/RECEIPT.md")
PY
ev "$(cat "$D/event.txt")"
