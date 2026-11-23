#!/bin/bash
# Score the CUDA_VISIBLE_DEVICES-fixed resubmissions (base x5, 8k u100/u150/u250) as each job ends.
set -u
O=/weka/oe-adapt-default/jacobm/olmicro
SP=/tmp/claude-0/-weka-oe-adapt-default-jacobm-olmicro-miles-olmo-core--claude-worktrees-bridge-cse-01TgX3YnCiqBAC4S2yxerMdT/a9337bf4-c3a0-51fb-a081-e1d79b9b351f/scratchpad
declare -A EXP=([base]=01M7SCABB46FF20NMDBW1PWDQ4 [8k]=01M7SCAEJXKGKDKJNMGEZ5127D)
declare -A TAGS=([base]="base-2.7b base-1.4b base-810m base-450m base-275m" [8k]="8k-u100 8k-u150 8k-u250")
declare -A DONE=()
DEADLINE=$(( $(date +%s) + 3*3600 ))
while [ ${#DONE[@]} -lt 2 ] && [ $(date +%s) -lt $DEADLINE ]; do
  for k in base 8k; do
    [ -n "${DONE[$k]:-}" ] && continue
    s=$(timeout 60 beaker experiment get ${EXP[$k]} --format json 2>/dev/null | python3 -c "
import json,sys; e=json.load(sys.stdin); e=e[0] if isinstance(e,list) else e; j=e['jobs'][-1]['status'] if e.get('jobs') else {}
print('ended', j.get('exitCode')) if j.get('exited') or j.get('canceled') else print('running')" 2>/dev/null)
    case "$s" in ended*)
      DONE[$k]=1; echo "$(date -u +%H:%M:%SZ) $k job $s"
      for tag in ${TAGS[$k]}; do
        out=$O/monitor/extra-evals-$tag-20261118; n=$(ls $out/*.raw.json 2>/dev/null | wc -l)
        if [ "$n" -eq 8 ]; then
          (cd $SP && PYTHONPATH=$O/open-instruct timeout 3600 $SP/hfenv/bin/python $O/tools/score_extra_evals.py $out $O/data/olmicro-extra-evals-r1-v1 $SP/repos/olmo-eval/src/olmo_eval/evals/extract/math.py --pool 32 >/dev/null 2>&1)
          python3 -c "import json; d=json.load(open('$out/summary.json')); m=d['minerva_math']; a=d['aime2026']; print('RESULT $tag minerva macro', m['macro_oe'], 'micro', m['acc_oe'], '| aime pass@1', a['pass@1_oe'], 'pass@32', a['pass@32_oe'], 'pass@1_16k', a['pass@1_oe_16k'], '| tokens', m['mean_tokens'], a['mean_tokens'], '| trunc', m['truncated_pct'], a['truncated_pct'])"
        else echo "$tag incomplete $n/8: $(tr '\n' ' ' < $out/exit-status 2>/dev/null)"; grep -hE "Error" $out/gpu*.log 2>/dev/null | tail -2; fi
      done;;
    esac
  done
  sleep 60
done
