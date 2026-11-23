#!/bin/bash
# Hand the 32k run from 4+4 (1 node) to 4+12 (2 nodes) at the update-100 boundary without exceeding 3 nodes:
# wait for the old run's checkpoint 99 AND its update-100 eval, stop it, then submit the miles.load resume run.
# Anything unexpected: log and exit without submitting.
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
    if r.get("rollout_id")==99 and r.get("stage")=="evaluation" and r.get("passed"): ev=True
    if r.get("rollout_id")==99 and r.get("stage")=="checkpoint" and r.get("passed"): ck=True
print(latest, ck, ev)
PY
}
while :; do
  read latest ck ev < <(state)
  if [ "$latest" = 99 ] && [ "$ck" = True ] && [ "$ev" = True ]; then log "checkpoint 99 committed and update-100 eval done"; break; fi
  if [ $(date +%s) -gt $DEADLINE ]; then log "DEADLINE: latest=$latest ck=$ck ev=$ev; not switching"; exit 2; fi
  st=$(timeout 60 beaker job get $OLD_JOB --format json 2>/dev/null | python3 -c "import json,sys; j=json.load(sys.stdin); j=j[0] if isinstance(j,list) else j; s=j['status']; print('ended' if s.get('exited') or s.get('canceled') else 'running')" 2>/dev/null)
  if [ "$st" = ended ]; then log "old job ended unexpectedly before the boundary; not switching"; exit 3; fi
  sleep 30
done
[ -f "$OLD_ROOT/checkpoints/rollout/global_dataset_state_dict_99.pt" ] || { log "cursor 99 missing; not switching"; exit 4; }
python3 - "$O" <<'PY'
import json,sys
from pathlib import Path
O=Path(sys.argv[1]); s=json.loads((O/"runs/olmicro-2.7b-lc-math-rl-v2-refresh-32k-20261118/checkpoints/driver_timing.jsonl").read_text().splitlines()[-1])
print("last old stage:", s)
PY
log "stopping old experiment $OLD_EXP"
timeout 120 beaker experiment stop $OLD_EXP >/dev/null 2>&1
for i in $(seq 1 20); do
  c=$(timeout 60 beaker job get $OLD_JOB --format json 2>/dev/null | python3 -c "import json,sys; j=json.load(sys.stdin); j=j[0] if isinstance(j,list) else j; print(bool(j['status'].get('canceled') or j['status'].get('exited')))" 2>/dev/null)
  [ "$c" = True ] && break; sleep 15
done
[ "$c" = True ] || { log "old job did not stop; not submitting"; exit 5; }
log "old job stopped; submitting 4+12 resume"
cd $O/open-instruct && PYTHONPATH=$PWD timeout 600 python3 ../tools/submit_excluding.py 01M6HMYBFW8D9ZK4SKGJ1XD6HW $NEW_TOML --exclude holmes-cs-aus-511.reviz.ai2.in > $O/monitor/v2-4x12-submit.json 2>&1 || { log "submit failed:"; cat $O/monitor/v2-4x12-submit.json; exit 6; }
cat $O/monitor/v2-4x12-submit.json
NEW_JOB=$(python3 -c "import json; print(json.load(open('$O/monitor/v2-4x12-submit.json'))['job'])")
echo "$NEW_JOB" > $O/monitor/v2-4x12-job-id
cp "$(python3 -c "import json; print(json.load(open('$O/monitor/v2-4x12-submit.json'))['receipt'])")" $O/monitor/v2-4x12-launch-receipt.json
nohup bash $O/tools/wait_job.sh $NEW_JOB 7100 60 >> $O/monitor/v2-4x12-status.log 2>&1 &
for i in $(seq 1 50); do
  s=$(timeout 60 beaker job get $NEW_JOB --format json 2>/dev/null | python3 -c "import json,sys; j=json.load(sys.stdin); j=j[0] if isinstance(j,list) else j; s=j['status']; print('started' if s.get('started') else ('ended' if s.get('exited') or s.get('canceled') else 'waiting'))" 2>/dev/null)
  [ "$s" != waiting ] && { log "new job $NEW_JOB $s"; exit 0; }
  sleep 30
done
log "new job $NEW_JOB NOT STARTED after 25 min"; exit 7
