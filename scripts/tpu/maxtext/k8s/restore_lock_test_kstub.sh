#!/bin/bash
# fake kubectl: state in $S/{active,admitted,applied}; admission happens only when active and $S/capacity exists
S=${STUB_DIR:?}; echo "K $*" >> $S/calls
case "$1" in
  kustomize) printf 'kind: ConfigMap\nmetadata:\n  name: x\n---\napiVersion: jobset.x-k8s.io/v1alpha2\nkind: JobSet\nmetadata:\n  name: tjs\n';;
  apply) touch $S/applied; echo true > $S/active; [ -f $S/capacity ] && touch $S/admitted; true;;
  get) if [ "$2" = workloads ]; then [ -f $S/applied ] && echo workload.kueue.x-k8s.io/jobset-tjs-abc12; else [ -f $S/admitted ] && echo True || echo False; fi;;
  patch) v=$(echo "$4$5$6" | grep -o 'true\|false'); echo $v > $S/active
         if [ $v = false ]; then rm -f $S/admitted; echo "evict-or-deactivate" >> $S/calls; else [ -f $S/capacity ] && touch $S/admitted; fi;;
esac
