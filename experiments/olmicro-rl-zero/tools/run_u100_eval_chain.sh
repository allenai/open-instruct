#!/bin/bash
# Update-100 eval for the 32k run: wait for the export job (checkpoint 99 -> HF), then submit the eval-only run, stop it
# after its initial eval, and print the eval metrics. Logs and exits on anything unexpected.
set -u
O=/weka/oe-adapt-default/jacobm/olmicro
EXP_ID=01M7RRPQ6XQ0ZHAN7PAK202GFZ
EXP_ROOT=$O/runs/olmicro-2.7b-lc-math-rl-v2-refresh-32k-u100-export-20261118
EV_TOML=runs/olmicro-2.7b-lc-math-rl-v2-refresh-32k-u100-eval-20261118.toml
EV_ROOT=$O/runs/olmicro-2.7b-lc-math-rl-v2-refresh-32k-u100-eval-20261118
DEADLINE=$(( $(date +%s) + ${1:-10800} ))
log(){ echo "$(date -u +%H:%M:%SZ) $*"; }
exp_jobs(){ timeout 60 beaker experiment get $1 --format json 2>/dev/null | python3 -c "
import json,sys; e=json.load(sys.stdin); e=e[0] if isinstance(e,list) else e
for j in e.get('jobs',[]):
    s=j['status']; print(j['id'], 'ended' if s.get('exited') or s.get('canceled') else ('started' if s.get('started') else 'waiting'), s.get('exitCode'))"; }
while :; do
  read jid st code < <(exp_jobs $EXP_ID | head -1)
  [ "$st" = ended ] && break
  [ $(date +%s) -gt $DEADLINE ] && { log "DEADLINE waiting for export ($st)"; exit 2; }
  sleep 30
done
log "export job $jid ended code=$code; workflow: $(python3 -c "import json; print(json.load(open('$EXP_ROOT/workflow.json')).get('status'))" 2>/dev/null)"
ls -la $EXP_ROOT/export-hf 2>&1 | head -20
[ "$code" = 0 ] && [ -f $EXP_ROOT/export-hf/config.json ] || { log "export failed or incomplete; not submitting eval"; timeout 120 beaker job logs $jid 2>/dev/null | grep -E "Error|Traceback|error" | tail -15; exit 3; }
cd $O/open-instruct
timeout 600 python3 ../tools/preflight.py $EV_TOML 2>&1 | tail -3
PYTHONPATH=$PWD timeout 600 python3 ../tools/submit_excluding.py 01M6HMYBFW8D9ZK4SKGJ1XD6HW $EV_TOML --exclude holmes-cs-aus-511.reviz.ai2.in > $O/monitor/u100-eval-submit.json 2>&1 || { log "eval submit failed"; cat $O/monitor/u100-eval-submit.json; exit 4; }
EV_ID=$(python3 -c "import json; print(json.load(open('$O/monitor/u100-eval-submit.json'))['url'].rsplit('/',1)[1])")
log "eval experiment https://beaker.org/ex/$EV_ID"
while :; do
  ev=$(python3 - "$EV_ROOT" <<'PY' 2>/dev/null
import json,sys
try:
    for l in open(sys.argv[1]+"/checkpoints/driver_timing.jsonl"):
        r=json.loads(l)
        if r.get("stage")=="evaluation" and r.get("rollout_id")==0: print("passed" if r.get("passed") else "failed"); break
    else: print("pending")
except FileNotFoundError: print("pending")
PY
)
  read ejid est ecode < <(exp_jobs $EV_ID | head -1)
  [ "$ev" != pending ] && { log "initial eval $ev"; break; }
  [ "$est" = ended ] && { log "eval job ended before eval (code=$ecode)"; break; }
  [ $(date +%s) -gt $DEADLINE ] && { log "DEADLINE waiting for eval"; break; }
  sleep 30
done
timeout 120 beaker experiment stop $EV_ID >/dev/null 2>&1; log "stopped eval experiment"
sleep 20
timeout 200 beaker job logs $ejid > $O/monitor/u100-eval-logs.txt 2>/dev/null
python3 - $O/monitor/u100-eval-logs.txt <<'PY'
import re,sys
text=open(sys.argv[1]).read()
print("EVAL u100:", {n: round(100*float(v),1) for n,v in re.findall(r"'eval/(math500|gsm8k_platinum)/episode_raw_reward': ([-0-9.e]+)", text)})
PY
