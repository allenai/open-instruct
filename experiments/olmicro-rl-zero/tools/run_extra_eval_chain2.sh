#!/bin/bash
# 8k u250: wait for its HF export, submit its eval; base models: one job. Score every output dir once its job ends.
set -u
O=/weka/oe-adapt-default/jacobm/olmicro
SP=/tmp/claude-0/-weka-oe-adapt-default-jacobm-olmicro-miles-olmo-core--claude-worktrees-bridge-cse-01TgX3YnCiqBAC4S2yxerMdT/a9337bf4-c3a0-51fb-a081-e1d79b9b351f/scratchpad
EXPORT=01M7S6698A58GYKDR280E47913; BASE=01M7S66WRNEQAT0JTT4E248147; EVAL8=""
DEADLINE=$(( $(date +%s) + 4*3600 ))
log(){ echo "$(date -u +%H:%M:%SZ) $*"; }
state(){ timeout 60 beaker experiment get $1 --format json 2>/dev/null | python3 -c "
import json,sys; e=json.load(sys.stdin); e=e[0] if isinstance(e,list) else e; j=e['jobs'][-1]['status'] if e.get('jobs') else {}
print('ended', j.get('exitCode')) if j.get('exited') or j.get('canceled') else print('running')" 2>/dev/null; }
score(){ local out=$1; local n=$(ls $out/*.raw.json 2>/dev/null | wc -l)
  if [ "$n" -eq 8 ]; then (cd $SP && PYTHONPATH=$O/open-instruct timeout 3600 $SP/hfenv/bin/python $O/tools/score_extra_evals.py $out $O/data/olmicro-extra-evals-r1-v1 $SP/repos/olmo-eval/src/olmo_eval/evals/extract/math.py --pool 32 >/dev/null 2>&1)
    python3 -c "import json; d=json.load(open('$out/summary.json')); m=d['minerva_math']; a=d['aime2026']; print('RESULT $(basename $out): minerva macro_oe', m['macro_oe'], 'micro_oe', m['acc_oe'], 'macro_mv', m['macro_mv'], '| aime pass@1_oe', a['pass@1_oe'], 'pass@32_oe', a['pass@32_oe'], 'pass@1_oe_16k', a['pass@1_oe_16k'], '| tokens', m['mean_tokens'], a['mean_tokens'], 'trunc', m['truncated_pct'], a['truncated_pct'])"
  else log "$(basename $out): only $n/8 shards"; grep -hE "Error|Traceback" $out/gpu*.log 2>/dev/null | tail -4; fi; }
base_done=""; e8_done=""
while [ $(date +%s) -lt $DEADLINE ]; do
  if [ -z "$EVAL8" ]; then
    s=$(state $EXPORT)
    case "$s" in ended*)
      W=$O/runs/olmicro-2.7b-lc-math-rl-v2-refresh-8k-u250-export-20261118
      if [ "$s" = "ended 0" ] && [ -f $W/export-hf/config.json ]; then
        EVAL8=$(timeout 120 beaker experiment create $O/tools/extra_evals_8k-u250_beaker.json --workspace ai2/olmo3p5-training --format json 2>/dev/null | python3 -c "import json,sys; e=json.load(sys.stdin); e=e[0] if isinstance(e,list) else e; print(e['id'])")
        log "8k u250 export done; eval submitted https://beaker.org/ex/$EVAL8"
      else log "8k u250 export failed ($s)"; EVAL8=failed; e8_done=1; fi;;
    esac
  elif [ -z "$e8_done" ]; then
    case "$(state $EVAL8)" in ended*) e8_done=1; score $O/monitor/extra-evals-8k-u250-20261118;; esac
  fi
  if [ -z "$base_done" ]; then
    case "$(state $BASE)" in ended*) base_done=1; for s in 2.7b 1.4b 810m 450m 275m; do score $O/monitor/extra-evals-base-$s-20261118; done;; esac
  fi
  [ -n "$base_done" ] && [ -n "$e8_done" ] && exit 0
  sleep 60
done
log "DEADLINE base=$base_done eval8=$EVAL8 done=$e8_done"
