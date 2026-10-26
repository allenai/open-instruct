#!/bin/bash
# H134 P2 prep, continuation after the upload (convert.sh's verify used a float byte sum and stopped at UPLOAD-MISMATCH;
# the upload itself completed). Re-verifies the GCS copy with exact sums, then the HF export and the tensor compare.
# (tpu-worker fork, 2026-10-26): two-stage OLMo-core -> MaxText convert of olmo35-small-14t600b-mtlc-emo
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
(cd $I/0 && find . -type f -printf '%P %s\n' | sort) > $W/items_files.txt
gn=$(gsutil ls -r "$G/0/**" 2>/dev/null | grep -c '^gs://'); gb=$(gsutil du -s $G/0 | awk '{print $1}')
ln=$(wc -l < $W/items_files.txt); lb=$(awk '{s+=$2} END {printf "%.0f", s}' $W/items_files.txt)
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
