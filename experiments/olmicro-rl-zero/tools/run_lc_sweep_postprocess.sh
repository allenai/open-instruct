#!/bin/bash
# Usage: run_lc_sweep_postprocess.sh SIZE EXPERIMENT_ID. Wait for the SIZE sweep, score it on CPU, report pass-rate bands for the 2.7B-solved pool vs the control, length/cap
# stats, and build the SIZE pass-band set at the 32k cap (plus 16k for comparison). Logs and exits on anything unexpected.
set -u
O=/weka/oe-adapt-default/jacobm/olmicro
SP=/tmp/claude-0/-weka-oe-adapt-default-jacobm-olmicro-miles-olmo-core--claude-worktrees-bridge-cse-01TgX3YnCiqBAC4S2yxerMdT/a9337bf4-c3a0-51fb-a081-e1d79b9b351f/scratchpad
SIZE=$1; EXP=$2; OUT=$O/monitor/passrate-sweep-lc$SIZE-32k-20261118; POOL=$O/data/dolci-math-base-r1-v1-sweep-pool-lc2.7b
DEADLINE=$(( $(date +%s) + 5*3600 ))
log(){ echo "$(date -u +%H:%M:%SZ) $*"; }
while :; do
  st=$(timeout 60 beaker experiment get $EXP --format json 2>/dev/null | python3 -c "
import json,sys; e=json.load(sys.stdin); e=e[0] if isinstance(e,list) else e; j=e['jobs'][-1]['status'] if e.get('jobs') else {}
print('ended', j.get('exitCode')) if j.get('exited') or j.get('canceled') else print('started' if j.get('started') else 'waiting')" 2>/dev/null)
  case "$st" in ended*) break;; esac
  [ $(date +%s) -gt $DEADLINE ] && { log "DEADLINE ($st, $(ls $OUT/*.raw.json 2>/dev/null | wc -l)/32 shards)"; exit 2; }
  sleep 60
done
n=$(ls $OUT/*.raw.json 2>/dev/null | wc -l); log "$SIZE sweep job $st; shards $n/32; $(tr '\n' ' ' < $OUT/exit-status 2>/dev/null)"
[ "$n" -eq 32 ] || { grep -hE "Error|Traceback" $OUT/gpu*.log | tail -8; exit 3; }
cd $SP && PYTHONPATH=$O/open-instruct timeout 3600 $SP/hfenv/bin/python $O/tools/score_format_probe.py $OUT 32 > $OUT/score.stdout 2>&1; log "scored: $(ls $OUT/*.scored.json | wc -l) files"
python3 - "$OUT" "$POOL" <<'PY'
import json, glob, sys, collections, statistics as st
out, pool = sys.argv[1], sys.argv[2]
kind = {json.loads(l)["metadata"]["prepared_sample_id"]: json.loads(l)["metadata"]["sweep_pool"] for l in open(pool + "/train.jsonl")}
by = collections.defaultdict(collections.Counter); toks = collections.defaultdict(list); cap = collections.Counter(); tot = collections.Counter()
for f in sorted(glob.glob(out + "/*.scored.json")):
    for row in json.load(open(f))["rows"]:
        k = kind[row["id"]]; c = sum(float(r) >= 1 for r in row["rewards"])
        by[k]["0/16" if c == 0 else ("15-16/16" if c >= 15 else "1-14/16")] += 1
        toks[k] += row["tokens"]; cap[k] += sum(t >= 32768 for t in row["tokens"]); tot[k] += len(row["tokens"])
for k in sorted(by):
    print(f"POOL {k}: prompts {sum(by[k].values())} bands {dict(by[k])} mean tokens {st.mean(toks[k]):.0f} at-32k-cap {100*cap[k]/tot[k]:.2f}%")
PY
for c in 32768 16384; do
  D=$O/data/dolci-math-base-r1-v1-passband-lc$SIZE-$((c/1024))k
  [ -e $D ] && { log "$D exists; skipping"; continue; }
  python3 $O/data-prep/filter_by_pass_rate.py $OUT $POOL $D --min-pass 0.0625 --max-pass 0.875 --min-samples 16 --unobserved drop --cap $c 2>&1 | python3 -c "import json,sys; d=json.load(sys.stdin); print('FILTER cap $c:', d['kept'], d['reasons'])"
done
PYTHONPATH=$O/open-instruct timeout 1200 $SP/hfenv/bin/python $O/tools/length_tail.py $OUT 16384 7680 2>&1 | tail -25
