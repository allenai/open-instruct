#!/bin/bash
# H134 B0, bf16 start (ledger-manager 2026-10-26): round the 14T fp32 masters to bf16 (RNE), stage-2 convert the exact fp32
# upcast with o35/ unchanged, upload no-clobber to h134-lc-step5961-bf16, HF-export the same rounded weights, compare the
# export with OLMo-core's HF export (diagnostic). The fp32 items at h134-lc-step5961 are kept. Writes only new paths.
set -uo pipefail
export CLOUDSDK_CONFIG=/tmp/gcloud-cfg CLOUDSDK_CORE_ACCOUNT=tpu-posttrain-agent@ai2-olmo.iam.gserviceaccount.com
W=/weka/oe-eval-default/abhishekr/watch/tpu/h134/convert-bf16; mkdir -p $W; cd $W
C=/weka/olmo-3p5-checkpoints/protected/olmo35-14t600b-mt100b-lc-sft-20261014/lc/step5961/olmo-core
JHF=/weka/olmo-3p5-checkpoints/scratch/olmo35-14t600b-mt100b-lc-sft-20261014/olmo35-14t600b-mt100b-lc-sft-20261014-lc/emo/step5961/hf
P=/weka/olmo-3p5-post-checkpoints/abhishekr/tpu-posttrain
M=$P/maxtext-stage/olmo35-small-14t600b-mtlc-emo-lc5961-masters-h134
MB=$P/maxtext-stage/olmo35-small-14t600b-mtlc-emo-lc5961-masters-bf16-h134
I=$P/maxtext-stage/olmo35-small-14t600b-mtlc-emo-lc5961-bf16-items-h134
HF=$P/hf/olmo35-small-14t600b-mtlc-emo-lc5961-bf16-base-h134
G=gs://ai2-olmo-dev/post-training/checkpoints/olmo35-small-14t600b-mtlc-emo/h134-lc-step5961-bf16
O35=/root/tpu-posttrain/o35; EXP=$P/export-job/export_maxtext_to_hf.py
OI=/root/repos/open-instruct/.venv/bin/python; MT=/tmp/h083mt/bin/python
SP=/root/repos/open-instruct/.venv/lib/python3.12/site-packages   # torch + safetensors for the exporter, beside h083mt's jax/orbax
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $W/convert.log >> /weka/oe-eval-default/abhishekr/watch/tpu/h134/events.log; }
sha256sum $W/round_bf16.py $O35/convert_olmocore_to_maxtext.py $EXP > $W/sources.sha256; log "bf16 sources: $(cut -c1-16 $W/sources.sha256 | tr '\n' ' ')"
gsutil ls $G/ >/dev/null 2>&1 && { log "GCS prefix $G exists; not overwriting"; exit 2; }
[ -e $M/DONE ] || { log "fp32 masters missing ($M)"; exit 3; }
[ -e $MB/DONE ] || { [ -e $MB ] && { log "$MB exists without DONE; not overwriting"; exit 4; }
  log "round start"; $OI $W/round_bf16.py $M $MB $W/bf16_manifest.json > $W/round.log 2>&1 || { log "ROUND-FAIL"; tail -5 $W/round.log; exit 5; }
  tail -1 $W/round.log > $MB/DONE; log "round: $(cat $MB/DONE)"; }
[ -e $I/0/items/commit_success.txt ] || { [ -e $I ] && { log "$I exists without commit marker; not overwriting"; exit 6; }
  log "stage 2 start"; JAX_PLATFORMS=cpu $MT $O35/convert_olmocore_to_maxtext.py $C $MB $I > $W/stage2.log 2>&1 || { log "STAGE2-FAIL"; tail -5 $W/stage2.log; exit 7; }
  log "stage 2: $(grep -E 'loaded|wrote' $W/stage2.log | tr '\n' ' ')"; }
(cd $I/0 && find . -type f -printf '%P %s\n' | sort) > $W/items_files.txt
ln=$(wc -l < $W/items_files.txt); lb=$(awk '{s+=$2} END {printf "%.0f", s}' $W/items_files.txt)
log "items: $ln files, $lb bytes; manifest sha $(sha256sum $I/0/items/manifest.ocdbt | cut -c1-16)"
gsutil ls $G/ >/dev/null 2>&1 && { log "GCS prefix $G appeared; not overwriting"; exit 8; }
log "upload start"; gsutil -m -q cp -n -r $I/0 $G/ > $W/upload.log 2>&1 || { log "UPLOAD-FAIL"; tail -5 $W/upload.log; exit 9; }
gn=$(gsutil ls -r "$G/0/**" 2>/dev/null | grep -c '^gs://'); gb=$(gsutil du -s $G/0 | awk '{printf "%.0f", $1}')
[ "$gn" = "$ln" ] && [ "$gb" = "$lb" ] && log "UPLOAD-OK $G/0/items ($gn files, $gb bytes)" || { log "UPLOAD-MISMATCH gcs $gn/$gb local $ln/$lb"; exit 10; }
[ -e $HF/model.safetensors.sha256 ] || { t=$HF.tmp; [ -e $t ] && { log "$t exists; not overwriting"; exit 11; }
  log "export start"; PYTHONPATH=$SP JAX_PLATFORMS=cpu $MT $EXP $I/0/items $JHF $t > $W/export.log 2>&1 && (cd $t && sha256sum model.safetensors > model.safetensors.sha256) && mv $t $HF \
    || { log "EXPORT-FAIL (hammond)"; tail -5 $W/export.log; exit 12; }
  log "EXPORT-OK $HF $(cut -c1-16 $HF/model.safetensors.sha256)"; }
PYTHONPATH=$SP $MT - "$HF/model.safetensors" "$JHF/model.safetensors" > $W/tensor_compare.json 2> $W/tensor_compare.err <<'PY'
import json, sys
from safetensors import safe_open
a, b = safe_open(sys.argv[1], "pt"), safe_open(sys.argv[2], "pt")
ka, kb = set(a.keys()), set(b.keys())
res = {"only_ours": sorted(ka - kb)[:20], "only_ref": sorted(kb - ka)[:20], "n_common": len(ka & kb), "max_abs": 0.0, "worst": None, "n_exact": 0, "shape_diff": [], "dtypes": {}}
for k in sorted(ka & kb):
    x, y = a.get_tensor(k), b.get_tensor(k)
    key = f"{x.dtype}|{y.dtype}"; res["dtypes"][key] = res["dtypes"].get(key, 0) + 1
    x, y = x.float(), y.float()
    if x.shape != y.shape: res["shape_diff"].append([k, list(x.shape), list(y.shape)]); continue
    d = float((x - y).abs().max()); res["n_exact"] += d == 0.0
    if d > res["max_abs"]: res["max_abs"], res["worst"] = d, k
json.dump(res, sys.stdout, indent=1)
PY
log "TENSOR-COMPARE bf16 export vs OLMo-core's HF export (diagnostic): $(python3 -c "import json;d=json.load(open('$W/tensor_compare.json'));print('common',d['n_common'],'exact',d['n_exact'],'only_ours',len(d['only_ours']),'only_ref',len(d['only_ref']),'shape_diff',len(d['shape_diff']),'max_abs',d['max_abs'],d['worst'],d['dtypes'])" 2>&1)"
log "BF16-CONVERT-DONE"
