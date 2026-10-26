#!/bin/bash
# H134: start prod.sh (it queues on H131's applies and the restore lock) once both HF step-1 references exist.
W=/weka/oe-eval-default/abhishekr/watch/tpu/h134
until [ -f $W/refce/refce_c11-s0.json ] && [ -f $W/refce/refce_c11-s1.json ]; do sleep 120; done
[ -f $W/prod/prod.pid ] && kill -0 $(cat $W/prod/prod.pid) 2>/dev/null && exit 0
cd $W && setsid -f nohup ./prod.sh >> prod/prod.out 2>&1 < /dev/null; sleep 3
ps -eo pid,args | awk '$2=="/bin/bash" && $3=="./prod.sh"{print $1}' | tail -1 > $W/prod/prod.pid
echo "$(date -u +%FT%TZ) prod.sh started by start_driver.sh, pid $(cat $W/prod/prod.pid)" >> $W/events.log
