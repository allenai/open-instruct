#!/bin/bash
# Poll a Beaker job until it exits, is canceled, or the deadline passes; print one status line per change.
# Usage: wait_job.sh JOB_ID [DEADLINE_SECONDS] [POLL_SECONDS]
job=$1; deadline=$(( $(date +%s) + ${2:-7200} )); poll=${3:-60}; last=""
while (( $(date +%s) < deadline )); do
  status=$(timeout 60 beaker job get "$job" --format json 2>/dev/null | python3 -c '
import json, sys
try:
    j = json.load(sys.stdin); j = j[0] if isinstance(j, list) else j; s = j.get("status", {})
except Exception:
    print("poll_error"); sys.exit()
if s.get("canceled"): print("canceled", s.get("canceledCode", ""))
elif s.get("exited"): print("exited", s.get("exitCode"))
elif s.get("started"): print("running")
elif s.get("scheduled"): print("scheduled")
else: print("queued")
')
  if [[ "$status" != "$last" && "$status" != "poll_error" ]]; then echo "$(date -u +%H:%M:%SZ) $job $status"; last=$status; fi
  case "$status" in exited*|canceled*) exit 0;; esac
  sleep "$poll"
done
echo "$(date -u +%H:%M:%SZ) $job deadline reached, last=$last"
