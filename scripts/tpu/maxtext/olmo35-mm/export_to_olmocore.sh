#!/bin/bash
# Convert one checkpoint of a TPU multimodal SFT run back to an OLMo-core DCP, for the GPU run's HF export and evals,
# and check the conversion bitwise. The eval half of ledger H271's pipeline (allenai/olmo-post-training-ledger@ca639ab0
# experiments/mmsft/tpu/run_tpu_eval.sh, stages stage, convert and verify).
#   export_to_olmocore.sh ITEMS DCP_OUT OLMO_CONFIG
#     ITEMS        <OUTPUT_DIR>/<RUN_NAME>/checkpoints/<S>/items on gs:// (staged first: params in full, only the
#                  metadata of the optimizer state), or a local copy staged before (with .STAGED_OK)
#     DCP_OUT      new dir for the OLMo-core DCP (model.* fp32 + config.json)
#     OLMO_CONFIG  an OLMo-core config.json of the recipe, written into the DCP for the exports (the reference run used
#                  its GPU twin's: H231 emo256k's step5481/config.json)
# MaxText's checkpoint dir S holds the state after S+1 optimizer updates: dir 5500 is OLMo-core step 5501, and the
# final save of a 10,962-step run is dir 10961 = step 10962. Name DCP_OUT by the OLMo-core step.
# Steps:
#   0 stage_items.sh (only for gs:// ITEMS) into STAGE_DIR (default /var/tmp/olmo35-mm-stage/<path>): waits for the
#     checkpoint's commit_success.txt, ~4 min for the ~48 GB of params. Needs a gcloud login.
#   1 the pinned converter, --mode maxtext_to_olmo_core, through fast_maxtext_loader.py (tensorstore reads; an Orbax
#     restore of a training checkpoint crawls and gets OOM-killed on a CPU box), validate=true: its own round trip
#     must report 0.0.
#   2 verify_roundtrip.py: the DCP read back and converted to the MaxText layout equals the MaxText params, bitwise
#     (every leaf, torch.equal). Writes DCP_OUT/ROUNDTRIP_BITWISE_OK and DCP_OUT/roundtrip.json.
# Env: PY, MAXTEXT_SRC as convert_b2e.sh; DRY_RUN=1 prints the commands only. Run one such job per box at a time.
set -euo pipefail
ITEMS=${1:?usage: export_to_olmocore.sh ITEMS DCP_OUT OLMO_CONFIG}
DCP=${2:?usage: export_to_olmocore.sh ITEMS DCP_OUT OLMO_CONFIG}
CFG=${3:?usage: export_to_olmocore.sh ITEMS DCP_OUT OLMO_CONFIG}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
DRY=${DRY_RUN:-}
ITEMS=${ITEMS%/}
if [[ -e "$DCP" ]]; then
  echo "$DCP exists; choose a new output dir (nothing is overwritten)" >&2
  exit 2
fi
if [[ -z "$DRY" && ! -f "$CFG" ]]; then
  echo "OLMo-core config $CFG not found" >&2
  exit 2
fi
export JAX_PLATFORMS=cpu
export PYTHONPATH="$HERE${MAXTEXT_SRC:+:$MAXTEXT_SRC}${PYTHONPATH:+:$PYTHONPATH}"
run() {
  echo "+ $*"
  [[ -n "$DRY" ]] || "$@"
}

LOCAL=$ITEMS
if [[ "$ITEMS" == gs://* ]]; then
  LOCAL=${STAGE_DIR:-/var/tmp/olmo35-mm-stage/${ITEMS#gs://}}
  echo "+ until gcloud storage objects describe $ITEMS/commit_success.txt; do sleep 300; done"
  if [[ -z "$DRY" ]]; then
    until gcloud storage objects describe "$ITEMS/commit_success.txt" >/dev/null 2>&1; do sleep 300; done
  fi
  run mkdir -p "$(dirname "$LOCAL")"
  run env PY="$PY" bash "$HERE/stage_items.sh" "$ITEMS" "$LOCAL"
elif [[ -z "$DRY" && ! -f "$LOCAL/.STAGED_OK" && ! -f "$LOCAL/commit_success.txt" ]]; then
  echo "$LOCAL is neither staged (.STAGED_OK) nor a committed checkpoint (commit_success.txt)" >&2
  exit 2
fi

run mkdir -p "$(dirname "$DCP")"
echo "+ $PY $HERE/fast_maxtext_loader.py --mode maxtext_to_olmo_core --maxtext_items_path $LOCAL --output_path $DCP" \
  "--olmo_config_path $CFG --validate true --save_overwrite false"
if [[ -z "$DRY" ]]; then
  log=$(mktemp "${TMPDIR:-/tmp}/olmo35-mm-convert.XXXXXX")
  "$PY" "$HERE/fast_maxtext_loader.py" --mode maxtext_to_olmo_core --maxtext_items_path "$LOCAL" --output_path "$DCP" \
    --olmo_config_path "$CFG" --validate true --save_overwrite false 2>&1 | tee "$log"
  grep -q "roundtrip_max_abs_diff': 0.0" "$log" || { echo "converter round trip not 0.0, see $log" >&2; exit 1; }
  printf '{"items": "%s", "olmo_core_config": "%s", "converted": "%s"}\n' "$ITEMS" "$CFG" "$(date -u +%FT%TZ)" \
    > "$DCP/TPU_SOURCE.json"
fi
echo "+ $PY $HERE/verify_roundtrip.py $DCP $LOCAL --out $DCP/roundtrip.json"
if [[ -z "$DRY" ]]; then
  "$PY" "$HERE/verify_roundtrip.py" "$DCP" "$LOCAL" --out "$DCP/roundtrip.json"
  touch "$DCP/ROUNDTRIP_BITWISE_OK"
fi
echo "OLMo-core DCP (bitwise round trip checked): $DCP"
