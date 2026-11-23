#!/bin/bash
# When the 1.4B RL run completes and exports, run the Minerva/AIME eval on its final export and score it.
set -u
O=/weka/oe-adapt-default/jacobm/olmicro
SP=/tmp/claude-0/-weka-oe-adapt-default-jacobm-olmicro-miles-olmo-core--claude-worktrees-bridge-cse-01TgX3YnCiqBAC4S2yxerMdT/a9337bf4-c3a0-51fb-a081-e1d79b9b351f/scratchpad
R=$O/runs/olmicro-1.4b-lc-math-rl-refresh-32k-20261118; OUT=$O/monitor/extra-evals-1.4b-final-20261118
log(){ echo "$(date -u +%H:%M:%SZ) $*"; }
for i in $(seq 1 150); do s=$(python3 -c "import json; print(json.load(open('$R/workflow.json')).get('status'))"); { [ "$s" = complete ] && [ -f $R/export-hf/.complete ]; } && break; [ "$s" = failed ] && { log "1.4B run FAILED"; python3 -c "import json; print(json.load(open('$R/workflow.json')).get('error','')[:600])"; exit 2; }; sleep 60; done
log "1.4B run $s; export ready"
F=$(ls $R/wandb/wandb/run-*/files/output_*.log | tail -1); python3 - "$F" <<'PY'
import re,sys
t=open(sys.argv[1],errors="ignore").read()
print("FINAL in-loop evals", [(n, round(100*float(v),1)) for n,v in re.findall(r"'eval/(math500|gsm8k_platinum)/episode_raw_reward': ([-0-9.e]+)", t)][-2:])
PY
E=$(timeout 120 beaker experiment create $O/tools/extra_evals_1.4b-final_beaker.json --workspace ai2/olmo3p5-training --format json 2>/dev/null | python3 -c "import json,sys; e=json.load(sys.stdin); e=e[0] if isinstance(e,list) else e; print(e['id'])")
log "eval submitted https://beaker.org/ex/$E"
for i in $(seq 1 120); do st=$(timeout 60 beaker experiment get $E --format json 2>/dev/null | python3 -c "
import json,sys; e=json.load(sys.stdin); e=e[0] if isinstance(e,list) else e; j=e['jobs'][-1]['status'] if e.get('jobs') else {}
print('ended') if j.get('exited') or j.get('canceled') else print('running')" 2>/dev/null); [ "$st" = ended ] && break; sleep 60; done
n=$(ls $OUT/*.raw.json 2>/dev/null | wc -l); log "eval ended; shards $n/8"
[ "$n" -eq 8 ] || { grep -hE "Error" $OUT/gpu*.log | tail -3; exit 3; }
cd $SP && PYTHONPATH=$O/open-instruct timeout 3600 $SP/hfenv/bin/python $O/tools/score_extra_evals.py $OUT $O/data/olmicro-extra-evals-r1-v1 $SP/repos/olmo-eval/src/olmo_eval/evals/extract/math.py --pool 32 >/dev/null 2>&1
python3 -c "import json; d=json.load(open('$OUT/summary.json')); m=d['minerva_math']; a=d['aime2026']; print('RESULT 1.4b-final minerva macro', m['macro_oe'], 'micro', m['acc_oe'], '| aime pass@1', a['pass@1_oe'], 'pass@32', a['pass@32_oe'], 'pass@1_16k', a['pass@1_oe_16k'], '| tokens', m['mean_tokens'], a['mean_tokens'], '| trunc', m['truncated_pct'], a['truncated_pct'])"
