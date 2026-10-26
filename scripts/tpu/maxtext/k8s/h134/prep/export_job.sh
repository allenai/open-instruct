#!/bin/bash
# H134 P2 prep, Beaker job: HF export of the converted 14T MaxText params (export-job/export_maxtext_to_hf.py, as H103's
# export_base.sh), then a diagnostic tensor compare against OLMo-core's HF export of lc/step5961. Hammond's venv install
# stalled (download.pytorch.org), so this step runs in a job, as export_base.sh did. Writes only new paths.
set -uo pipefail
W=/weka/oe-eval-default/abhishekr/watch/tpu/h134/convert
P=/weka/olmo-3p5-post-checkpoints/abhishekr/tpu-posttrain; EXP=$P/export-job/export_maxtext_to_hf.py
I=$P/maxtext-stage/olmo35-small-14t600b-mtlc-emo-lc5961-items-h134
HF=$P/hf/olmo35-small-14t600b-mtlc-emo-lc5961-base-h134; t=$HF.tmp-$BEAKER_JOB_ID
JHF=/weka/olmo-3p5-checkpoints/scratch/olmo35-14t600b-mt100b-lc-sft-20261014/olmo35-14t600b-mt100b-lc-sft-20261014-lc/emo/step5961/hf
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $W/convert.log; }
command -v uv || { python3 -m pip install -q uv || curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
uv venv /tmp/v --python 3.12 && uv pip install --python /tmp/v/bin/python --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0+cpu" \
  && uv pip install --python /tmp/v/bin/python "numpy==2.3.5" jax orbax-checkpoint safetensors || { log "VENV-FAIL (job $BEAKER_JOB_ID)"; exit 2; }
sha256sum $EXP
[ -f $HF/model.safetensors.sha256 ] && { log "export exists: $HF"; } || {
  [ -e $HF ] && { log "$HF exists without sha; not overwriting"; exit 3; }
  log "export start (job $BEAKER_JOB_ID)"
  JAX_PLATFORMS=cpu /tmp/v/bin/python $EXP $I/0/items $JHF $t > $W/export.log 2>&1 && (cd $t && sha256sum model.safetensors > model.safetensors.sha256) && mv $t $HF \
    || { log "EXPORT-FAIL (job $BEAKER_JOB_ID)"; tail -20 $W/export.log; exit 4; }
  log "EXPORT-OK $HF $(cut -c1-16 $HF/model.safetensors.sha256)"; }
/tmp/v/bin/python - "$HF/model.safetensors" "$JHF/model.safetensors" > $W/tensor_compare.json 2> $W/tensor_compare.err <<'PY'
import json, sys
from safetensors import safe_open
a, b = safe_open(sys.argv[1], "pt"), safe_open(sys.argv[2], "pt")
ka, kb = set(a.keys()), set(b.keys())
res = {"only_ours": sorted(ka - kb)[:20], "only_ref": sorted(kb - ka)[:20], "n_common": len(ka & kb), "max_abs": 0.0, "worst": None, "shape_diff": [], "dtypes": {}}
for k in sorted(ka & kb):
    x, y = a.get_tensor(k), b.get_tensor(k)
    res["dtypes"].setdefault(f"{x.dtype}|{y.dtype}", 0); res["dtypes"][f"{x.dtype}|{y.dtype}"] += 1
    x, y = x.float(), y.float()
    if x.shape != y.shape: res["shape_diff"].append([k, list(x.shape), list(y.shape)]); continue
    d = float((x - y).abs().max())
    if d > res["max_abs"]: res["max_abs"], res["worst"] = d, k
json.dump(res, sys.stdout, indent=1)
PY
log "TENSOR-COMPARE (diagnostic, vs OLMo-core's HF export): $(python3 -c "import json;d=json.load(open('$W/tensor_compare.json'));print('common',d['n_common'],'only_ours',len(d['only_ours']),'only_ref',len(d['only_ref']),'shape_diff',len(d['shape_diff']),'max_abs',d['max_abs'],d['worst'],d['dtypes'])" 2>&1)"
log "CONVERT-DONE"
