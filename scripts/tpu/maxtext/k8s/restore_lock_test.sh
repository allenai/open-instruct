# Stub test for restore_lock.sh (Linux, needs flock): bash restore_lock_test.sh. Checks that a workload is never active
# without the lock, the lock is free while it waits inactive, apply waits for another holder, and big spacing holds.
set -u
T=$(mktemp -d); export STUB_DIR=$T/s; mkdir -p $STUB_DIR
export RL_DIR=$T RL_K="bash $(dirname $0)/restore_lock_test_kstub.sh" RL_ADMIT_WAIT=20 RL_RETRY=5
echo free > $T/restore.lock
source $(dirname $0)/restore_lock.sh
# case 1: no capacity at first; capacity arrives after ~30 s; lock must be free while inactive
( sleep 32; touch $STUB_DIR/capacity ) &
( for i in $(seq 1 30); do read -r h t < $T/restore.lock; a=$(cat $STUB_DIR/active 2>/dev/null); echo "$(date +%s) lock=$h active=$a" >> $T/trace; sleep 2; done ) &
restore_lock_apply tester /dummy; rc=$?
read -r h t < $T/restore.lock
echo "X1 rc=$rc holder=$h admitted=$([ -f $STUB_DIR/admitted ] && echo y || echo n)"
viol=$(awk '$3=="active=true" && $2!="lock=tester"' $T/trace | wc -l); echo "X1 active-without-lock samples: $viol"
grep -c 'lock=free active=false' $T/trace | sed 's/^/X1 free-while-inactive samples: /'
restore_lock_release tester; read -r h t < $T/restore.lock; echo "X1 after release: $h"
# case 2: other holder holds lock; apply must not happen until it is released
rm -rf $STUB_DIR; mkdir -p $STUB_DIR; touch $STUB_DIR/capacity
echo "other $(date +%s)" > $T/restore.lock
( sleep 10; echo free > $T/restore.lock ) &
t0=$(date +%s); restore_lock_apply tester /dummy; t1=$(date +%s)
first_apply=$(grep -n 'K apply' $STUB_DIR/calls | head -1 | cut -d: -f1)
echo "X2 waited $((t1-t0))s for other holder (>=10 expected); applied=$([ -n "$first_apply" ] && echo y)"
# case 3: big spacing
rm -rf $STUB_DIR; mkdir -p $STUB_DIR; touch $STUB_DIR/capacity; echo free > $T/restore.lock
echo $(( $(date +%s) - 1800 + 8 )) > $T/big_restore_end
t0=$(date +%s); restore_lock_apply tester2 /dummy big; t1=$(date +%s); echo "X3 big spacing wait $((t1-t0))s (>=8 expected)"
tail -12 $T/restore.lock.log | sed 's/^/X LOG /'
