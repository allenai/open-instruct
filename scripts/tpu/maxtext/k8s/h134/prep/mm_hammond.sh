#!/bin/bash
# H134 B1/B2 on Hammond (tpu-worker fork, 2026-10-26; ledger-manager: use the OLMo-core step exports, not the
# mid-checkpoints DCP, which Hammond doesn't mount and which was not read). Per base, in order:
#  1. sha256 receipts of the export (model.safetensors, olmo_core_config.json);
#  2. the verified base copy to /weka/olmo-3p5-post-checkpoints/bases/<registry name>/step<N>/ with RECEIPT.json
#     (no-clobber; source and copy sha256 compared; deletes nothing);
#  3. strip_vision.py plan + apply on the copy: the 415 LM tensors under B0's names as flat fp32 .npy (exact bf16
#     upcasts), every vocab row passed through (ledger-manager's pad-row ruling), vision/connector keys dropped.
set -uo pipefail
J=/weka/oe-eval-default/abhishekr/watch/tpu/h134/mmstrip; CODE=$J/code
E=/weka/oe-training-default/jasonr/olmo35-vision-alignment/exports
BASES=/weka/olmo-3p5-post-checkpoints/bases
STAGE=/weka/olmo-3p5-post-checkpoints/abhishekr/tpu-posttrain/maxtext-stage
PY=/root/repos/open-instruct/.venv/bin/python
R=$J/hammond-run; mkdir -p $R
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $R/run.log; }
declare -A SRC=([olmo35-small-mm-ckpt1-64k]=$E/olmoe3-810m-cx2-mm-merged-mt-lc-v6-lm1e4/step20000
                [olmo35-small-mm-ckpt2-256k-amt]=$E/olmoe3-810m-cx2-mm-hero-mt-lc-256k-lm1e4/step22240)
sha256sum $CODE/strip_vision.py $CODE/mm_hammond.sh $J/b0_lm_manifest.json $J/b0_config.json > $R/code.sha256
fail=0
for name in olmo35-small-mm-ckpt1-64k olmo35-small-mm-ckpt2-256k-amt; do
  src=${SRC[$name]}; step=$(basename $src); D=$BASES/$name/$step; W=$R/$name; mkdir -p $W
  if [ ! -e $D/RECEIPT.json ]; then
    [ -e $D ] && { log "[$name] $D exists without RECEIPT.json; not overwriting"; fail=1; continue; }
    log "[$name] receipt + copy $src -> $D"
    mkdir -p $D.tmp
    (cd $src && find . -type f -printf '%P\n' | sort) > $W/files.txt
    while read -r f; do
      s=$(sha256sum "$src/$f" | cut -d' ' -f1); mkdir -p "$(dirname "$D.tmp/$f")"
      cp -n "$src/$f" "$D.tmp/$f" || { log "[$name] COPY-FAIL $f"; fail=1; break; }
      d=$(sha256sum "$D.tmp/$f" | cut -d' ' -f1); printf '%s\t%s\t%s\t%s\n' "$f" "$s" "$d" "$(stat -c %s "$src/$f")"
    done < $W/files.txt > $W/copy.tsv
    nb=$(awk -F'\t' '$2!=$3' $W/copy.tsv | wc -l); n=$(wc -l < $W/copy.tsv); ns=$(wc -l < $W/files.txt)
    if [ "$nb" = 0 ] && [ "$n" = "$ns" ]; then
      $PY - $W/copy.tsv $src $D.tmp/RECEIPT.json <<'PY'
import json, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
json.dump({"source": sys.argv[2], "source_format": "OLMo-core weights-only export (model.safetensors + olmo_core_config.json)",
           "not_read": "the mid-checkpoints DCP (olmo-3p5-mid-checkpoints) is not mounted on Hammond and was not read",
           "n_files": len(rows), "bytes": sum(int(r[3]) for r in rows),
           "files": {r[0]: {"sha256": r[1], "bytes": int(r[3])} for r in rows}}, open(sys.argv[3], "w"), indent=1)
PY
      mv $D.tmp $D && log "[$name] COPY-OK $D: $n files, RECEIPT.json $(sha256sum $D/RECEIPT.json | cut -c1-16)"
    else log "[$name] COPY-FAIL: $n/$ns files, $nb sha mismatches; partial copy kept at $D.tmp"; fail=1; continue; fi
  else log "[$name] $D/RECEIPT.json exists; using the verified copy"; fi
  out=$STAGE/$name-$step-masters-stripped-h134
  $PY $CODE/strip_vision.py plan --ref $J/b0_lm_manifest.json --ref-config $J/b0_config.json --src $D \
      --src-safetensors $D/model.safetensors --src-config $D/olmo_core_config.json --lm-prefix lm. --lm-config-key model.lm \
      --mm-tokenizer-size 100278 --allow-bf16-upcast --pad-rows passthrough --out $W/plan.json >> $R/run.log 2>&1 \
    || { log "[$name] PLAN-FAIL: $(tail -1 $R/run.log)"; fail=1; continue; }
  $PY $CODE/strip_vision.py apply --plan $W/plan.json --masters-in /nonexistent --masters-out $out --receipt $W/strip_receipt.json >> $R/run.log 2>&1 \
    || { log "[$name] APPLY-FAIL: $(tail -1 $R/run.log)"; fail=1; continue; }
  log "[$name] STRIP-OK $out"
done
log "RUN-DONE fail=$fail"
