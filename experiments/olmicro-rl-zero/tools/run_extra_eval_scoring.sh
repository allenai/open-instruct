#!/bin/bash
# Wait for the two extra-eval Beaker jobs, then score each finished one on CPU (both graders) and print summaries.
set -u
O=/weka/oe-adapt-default/jacobm/olmicro
SP=/tmp/claude-0/-weka-oe-adapt-default-jacobm-olmicro-miles-olmo-core--claude-worktrees-bridge-cse-01TgX3YnCiqBAC4S2yxerMdT/a9337bf4-c3a0-51fb-a081-e1d79b9b351f/scratchpad
declare -A EXP=([8k-final]=01M7S4FN0MQJPJAB0ERBP3QMX7 [32k-u250]=01M7S4FRAT4Z92M3YBK0YQ1WRV)
declare -A DONE=()
DEADLINE=$(( $(date +%s) + 3*3600 ))
while [ ${#DONE[@]} -lt 2 ] && [ $(date +%s) -lt $DEADLINE ]; do
  for tag in 8k-final 32k-u250; do
    [ -n "${DONE[$tag]:-}" ] && continue
    st=$(timeout 60 beaker experiment get ${EXP[$tag]} --format json 2>/dev/null | python3 -c "
import json,sys; e=json.load(sys.stdin); e=e[0] if isinstance(e,list) else e; j=e['jobs'][-1]['status'] if e.get('jobs') else {}
print('ended', j.get('exitCode')) if j.get('exited') or j.get('canceled') else print('running')" 2>/dev/null)
    case "$st" in ended*)
      DONE[$tag]=1; OUT=$O/monitor/extra-evals-$tag-20261118
      echo "$(date -u +%H:%M:%SZ) $tag job $st; shards: $(ls $OUT/*.raw.json 2>/dev/null | wc -l)/8; $(tr '\n' ' ' < $OUT/exit-status 2>/dev/null)"
      if [ $(ls $OUT/*.raw.json 2>/dev/null | wc -l) -eq 8 ]; then
        (cd $SP && PYTHONPATH=$O/open-instruct timeout 3600 $SP/hfenv/bin/python $O/tools/score_extra_evals.py $OUT $O/data/olmicro-extra-evals-r1-v1 $SP/repos/olmo-eval/src/olmo_eval/evals/extract/math.py --pool 32 2>/dev/null > $OUT/summary.stdout)
        echo "== $tag summary"; cat $OUT/summary.json
      else
        echo "$tag incomplete"; grep -hE "Error|Traceback" $OUT/gpu*.log 2>/dev/null | tail -5
      fi;;
    esac
  done
  sleep 60
done
