#!/bin/bash
# v2 handoff (node limit relaxed): submit the 4+12 miles.load resume as soon as the old run's checkpoint 99 is
# committed, then stop the old run once its update-100 eval is done. If monitor/fla-fix-failed exists at submit
# time, strip the FLA config-cache env block from the new TOML first. Unexpected states: log and exit.
set -u
O=/weka/oe-adapt-default/jacobm/olmicro
OLD_ROOT=$O/runs/olmicro-2.7b-lc-math-rl-v2-refresh-32k-20261118
OLD_EXP=01M7R6Z11XWGR46DA7TWM8HDH5; OLD_JOB=01M7R6Z15JPX5VC6P9R4R3JWM0
NEW_TOML=runs/olmicro-2.7b-lc-math-rl-v2-refresh-32k-4x12-20261118.toml
DEADLINE=$(( $(date +%s) + ${1:-5400} ))
log(){ echo "$(date -u +%H:%M:%SZ) $*"; }
state(){ python3 - "$OLD_ROOT" <<'PY'
import json,sys
root=sys.argv[1]
try: latest=json.load(open(root+"/checkpoints/core-latest.json"))["rollout_id"]
except Exception: latest=None
ev=ck=False
for line in open(root+"/checkpoints/driver_timing.jsonl"):
    try: r=json.loads(line)
    except Exception: continue
    if r.get("rollout_id")==99 and r.get("passed"):
        ev |= r.get("stage")=="evaluation"; ck |= r.get("stage")=="checkpoint"
print(latest, ck, ev)
PY
}
jobstate(){ timeout 60 beaker job get $1 --format json 2>/dev/null | python3 -c "import json,sys; j=json.load(sys.stdin); j=j[0] if isinstance(j,list) else j; s=j['status']; print('ended' if s.get('exited') or s.get('canceled') else ('started' if s.get('started') else 'waiting'))" 2>/dev/null; }
while :; do
  read latest ck ev < <(state)
  [ "$latest" = 99 ] && [ "$ck" = True ] && [ -f "$OLD_ROOT/checkpoints/rollout/global_dataset_state_dict_99.pt" ] && { log "checkpoint 99 committed (eval done: $ev)"; break; }
  [ $(date +%s) -gt $DEADLINE ] && { log "DEADLINE before checkpoint 99: latest=$latest; not switching"; exit 2; }
  [ "$(jobstate $OLD_JOB)" = ended ] && { log "old job ended before checkpoint 99; not switching"; exit 3; }
  sleep 20
done
cd $O/open-instruct
if [ -f $O/monitor/fla-fix-failed ]; then
  python3 - "$NEW_TOML" <<'PY'
import sys,re
p=sys.argv[1]; s=open(p).read()
s=re.sub(r'# FLA config cache .*?TRITON_PRINT_AUTOTUNING = "1"\n', '', s, flags=re.S); open(p,"w").write(s)
PY
  log "FLA fix flagged failed: env block removed"
fi
grep -q FLA_CACHE_MODE $NEW_TOML && log "submitting WITH FLA config cache" || log "submitting WITHOUT FLA config cache"
PYTHONPATH=$PWD timeout 600 python3 ../tools/submit_excluding.py 01M6HMYBFW8D9ZK4SKGJ1XD6HW $NEW_TOML --exclude holmes-cs-aus-511.reviz.ai2.in > $O/monitor/v2-4x12-submit.json 2>&1 || { log "submit failed:"; cat $O/monitor/v2-4x12-submit.json; exit 6; }
cat $O/monitor/v2-4x12-submit.json
NEW_JOB=$(python3 -c "import json; print(json.load(open('$O/monitor/v2-4x12-submit.json'))['job'])")
echo "$NEW_JOB" > $O/monitor/v2-4x12-job-id
cp "$(python3 -c "import json; print(json.load(open('$O/monitor/v2-4x12-submit.json'))['receipt'])")" $O/monitor/v2-4x12-launch-receipt.json
nohup bash $O/tools/wait_job.sh $NEW_JOB 7100 60 >> $O/monitor/v2-4x12-status.log 2>&1 &
started=no; stopped=no
for i in $(seq 1 120); do
  if [ $started = no ]; then s=$(jobstate $NEW_JOB); [ "$s" != waiting ] && [ -n "$s" ] && { log "new job $NEW_JOB $s"; started=$s; }; fi
  if [ $stopped = no ]; then
    read latest ck ev < <(state)
    if [ "$ev" = True ] || [ "$(jobstate $OLD_JOB)" = ended ]; then
      log "old update-100 eval done ($ev); stopping old experiment $OLD_EXP"; timeout 120 beaker experiment stop $OLD_EXP >/dev/null 2>&1; stopped=yes
    fi
  fi
  [ $started != no ] && [ $stopped = yes ] && { log "handoff complete"; exit 0; }
  sleep 30
done
log "TIMEOUT: new started=$started old stopped=$stopped"; exit 7
