#!/bin/bash
# H134 P2 prep (tpu-worker fork, 2026-10-26): two-stage OLMo-core -> MaxText convert of olmo35-small-14t600b-mtlc-emo
# (registry checkpoint lc/step5961) with the converter used for olmo35-small-4t-emo-step5961 (o35/, unchanged), upload
# to a new GCS prefix (no-clobber), then the HF export of the converted params for the step-1 CE reference
# (export-job/export_maxtext_to_hf.py, as H103/H123's base-h103), plus a diagnostic tensor compare against the
# OLMo-core HF export of the same checkpoint. Writes only new paths; deletes nothing.
set -uo pipefail
export CLOUDSDK_CONFIG=/tmp/gcloud-cfg CLOUDSDK_CORE_ACCOUNT=tpu-posttrain-agent@ai2-olmo.iam.gserviceaccount.com
W=/weka/oe-eval-default/abhishekr/watch/tpu/h134/convert; mkdir -p $W; cd $W
C=/weka/olmo-3p5-checkpoints/protected/olmo35-14t600b-mt100b-lc-sft-20261014/lc/step5961/olmo-core
JHF=/weka/olmo-3p5-checkpoints/scratch/olmo35-14t600b-mt100b-lc-sft-20261014/olmo35-14t600b-mt100b-lc-sft-20261014-lc/emo/step5961/hf
P=/weka/olmo-3p5-post-checkpoints/abhishekr/tpu-posttrain
M=$P/maxtext-stage/olmo35-small-14t600b-mtlc-emo-lc5961-masters-h134
I=$P/maxtext-stage/olmo35-small-14t600b-mtlc-emo-lc5961-items-h134
HF=$P/hf/olmo35-small-14t600b-mtlc-emo-lc5961-base-h134
G=gs://ai2-olmo-dev/post-training/checkpoints/olmo35-small-14t600b-mtlc-emo/h134-lc-step5961
O35=/root/tpu-posttrain/o35; EXP=$P/export-job/export_maxtext_to_hf.py
OI=/root/repos/open-instruct/.venv/bin/python; MT=/tmp/h083mt/bin/python
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $W/convert.log; }
sha256sum $O35/dump_olmocore_masters.py $O35/convert_olmocore_to_maxtext.py $EXP $C/config.json > $W/sources.sha256
log "sources: $(cut -c1-16 $W/sources.sha256 | tr '\n' ' ')"
gsutil ls $G/ >/dev/null 2>&1 && { log "GCS prefix $G exists; not overwriting"; exit 2; }
[ -e $I/0/items/commit_success.txt ] || {
  [ -e $M/DONE ] || { log "stage 1 start"; $OI $O35/dump_olmocore_masters.py $C $M > $W/stage1.log 2>&1 || { log "STAGE1-FAIL"; tail -5 $W/stage1.log; exit 3; }
    tail -1 $W/stage1.log > $M/DONE; log "stage 1: $(cat $M/DONE)"; }
  log "stage 2 start"; [ -e $I ] && { log "items dir exists without commit marker; not overwriting"; exit 4; }
  JAX_PLATFORMS=cpu $MT $O35/convert_olmocore_to_maxtext.py $C $M $I > $W/stage2.log 2>&1 || { log "STAGE2-FAIL"; tail -5 $W/stage2.log; exit 5; }
  log "stage 2: $(grep -E 'loaded|wrote' $W/stage2.log | tr '\n' ' ')"; }
(cd $I/0 && find . -type f -printf '%P %s\n' | sort) > $W/items_files.txt
log "items: $(wc -l < $W/items_files.txt) files, $(awk '{s+=$2} END {print s}' $W/items_files.txt) bytes; manifest sha $(sha256sum $I/0/items/manifest.ocdbt | cut -c1-16)"
gsutil ls $G/ >/dev/null 2>&1 && { log "GCS prefix $G appeared; not overwriting"; exit 6; }
log "upload start"; gsutil -m -q cp -n -r $I/0 $G/ > $W/upload.log 2>&1 || { log "UPLOAD-FAIL"; tail -5 $W/upload.log; exit 7; }
gn=$(gsutil ls -r "$G/0/**" 2>/dev/null | grep -c '^gs://'); gb=$(gsutil du -s $G/0 | awk '{print $1}')
ln=$(wc -l < $W/items_files.txt); lb=$(awk '{s+=$2} END {print s}' $W/items_files.txt)
[ "$gn" = "$ln" ] && [ "$gb" = "$lb" ] && log "UPLOAD-OK $G/0/items ($gn files, $gb bytes)" || { log "UPLOAD-MISMATCH gcs $gn/$gb local $ln/$lb"; exit 8; }
[ -e $HF/model.safetensors.sha256 ] || {
  [ -x /tmp/h134exp/bin/python ] || { uv venv -q /tmp/h134exp --python 3.12 && uv pip install -q --python /tmp/h134exp/bin/python --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0+cpu" \
    && uv pip install -q --python /tmp/h134exp/bin/python "numpy==2.3.5" jax orbax-checkpoint safetensors; } || { log "VENV-FAIL"; exit 9; }
  t=$HF.tmp; [ -e $t ] && { log "$t exists; not overwriting"; exit 10; }
  log "export start"; JAX_PLATFORMS=cpu /tmp/h134exp/bin/python $EXP $I/0/items $JHF $t > $W/export.log 2>&1 && (cd $t && sha256sum model.safetensors > model.safetensors.sha256) && mv $t $HF \
    || { log "EXPORT-FAIL"; tail -5 $W/export.log; exit 11; }
  log "EXPORT-OK $HF $(cut -c1-16 $HF/model.safetensors.sha256)"; }
/tmp/h134exp/bin/python - "$HF/model.safetensors" "$JHF/model.safetensors" > $W/tensor_compare.json 2> $W/tensor_compare.err <<'PY'
import json, sys
from safetensors import safe_open
a, b = safe_open(sys.argv[1], "pt"), safe_open(sys.argv[2], "pt")
ka, kb = set(a.keys()), set(b.keys())
res = {"only_ours": sorted(ka - kb)[:20], "only_ref": sorted(kb - ka)[:20], "n_common": len(ka & kb), "max_abs": 0.0, "worst": None, "shape_diff": []}
for k in sorted(ka & kb):
    x, y = a.get_tensor(k).float(), b.get_tensor(k).float()
    if x.shape != y.shape: res["shape_diff"].append([k, list(x.shape), list(y.shape)]); continue
    d = float((x - y).abs().max())
    if d > res["max_abs"]: res["max_abs"], res["worst"] = d, k
json.dump(res, sys.stdout, indent=1)
PY
log "TENSOR-COMPARE (diagnostic, vs OLMo-core's HF export): $(python3 -c "import json;d=json.load(open('$W/tensor_compare.json'));print('common',d['n_common'],'only_ours',len(d['only_ours']),'only_ref',len(d['only_ref']),'shape_diff',len(d['shape_diff']),'max_abs',d['max_abs'],d['worst'])" 2>&1)"
log "CONVERT-DONE"
