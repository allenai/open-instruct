#!/bin/bash
# Usage: run_lc_rest_postprocess.sh SIZE EXPERIMENT_ID. Wait for the SIZE rest-of-pool sweep, score it, report its bands,
# then filter over all 13,251 prompts from both sweeps (pool + rest) at the 32k and 16k caps -> passband-lc<SIZE>-{32k,16k}-full.
set -u
O=/weka/oe-adapt-default/jacobm/olmicro
SP=/tmp/claude-0/-weka-oe-adapt-default-jacobm-olmicro-miles-olmo-core--claude-worktrees-bridge-cse-01TgX3YnCiqBAC4S2yxerMdT/a9337bf4-c3a0-51fb-a081-e1d79b9b351f/scratchpad
SIZE=$1; EXP=$2; REST=$O/monitor/passrate-sweep-lc$SIZE-32k-rest-20261118; POOLRUN=$O/monitor/passrate-sweep-lc$SIZE-32k-20261118
COMB=$O/monitor/passrate-sweep-lc$SIZE-32k-combined-20261118
DEADLINE=$(( $(date +%s) + 5*3600 ))
log(){ echo "$(date -u +%H:%M:%SZ) $*"; }
while :; do
  st=$(timeout 60 beaker experiment get $EXP --format json 2>/dev/null | python3 -c "
import json,sys; e=json.load(sys.stdin); e=e[0] if isinstance(e,list) else e; j=e['jobs'][-1]['status'] if e.get('jobs') else {}
print('ended', j.get('exitCode')) if j.get('exited') or j.get('canceled') else print('running')" 2>/dev/null)
  case "$st" in ended*) break;; esac
  [ $(date +%s) -gt $DEADLINE ] && { log "DEADLINE ($(ls $REST/*.raw.json 2>/dev/null | wc -l)/32)"; exit 2; }
  sleep 60
done
n=$(ls $REST/*.raw.json 2>/dev/null | wc -l); log "$SIZE rest sweep $st; shards $n/32; $(tr '\n' ' ' < $REST/exit-status 2>/dev/null)"
[ "$n" -eq 32 ] || { grep -hE "Error|Traceback" $REST/gpu*.log | tail -8; exit 3; }
cd $SP && PYTHONPATH=$O/open-instruct timeout 3600 $SP/hfenv/bin/python $O/tools/score_format_probe.py $REST 32 > $REST/score.stdout 2>&1; log "scored: $(ls $REST/*.scored.json | wc -l)"
mkdir -p $COMB
for f in $POOLRUN/*.scored.json; do ln -sfn "$f" "$COMB/pool_$(basename "$f")"; done
for f in $REST/*.scored.json; do ln -sfn "$f" "$COMB/rest_$(basename "$f")"; done
log "combined scored files: $(ls $COMB/*.scored.json | wc -l)"
python3 - "$REST" <<'PY'
import json, glob, sys, collections
c = collections.Counter()
for f in glob.glob(sys.argv[1] + "/*.scored.json"):
    for row in json.load(open(f))["rows"]: c[sum(float(r) >= 1 for r in row["rewards"])] += 1
inb = sum(v for k, v in c.items() if 1 <= k <= 14)
print(f"REST: prompts {sum(c.values())}, in band {inb} ({100*inb/sum(c.values()):.1f}%), at 1/16 {c[1]}, >=15/16 {sum(v for k,v in c.items() if k>=15)}")
PY
for cap in 32768 16384; do
  D=$O/data/dolci-math-base-r1-v1-passband-lc$SIZE-$((cap/1024))k-full
  [ -e $D ] && { log "$D exists; skipping"; continue; }
  python3 $O/data-prep/filter_by_pass_rate.py $COMB $O/data/dolci-math-base-r1-v1 $D --min-pass 0.0625 --max-pass 0.875 --min-samples 16 --unobserved drop --cap $cap 2>&1 | python3 -c "import json,sys; d=json.load(sys.stdin); print('FILTER full cap $cap:', d['kept'], d['reasons'], 'observed', d['observed_prompts'])"
done
