#!/bin/bash
# Convert an OLMo-core multimodal checkpoint (e.g. B2e step 22240) to the MaxText Orbax items that
# ../k8s/olmo35-mm-sft-4x4x8 starts from, seeding the <think>/</think> rows on the way, then check the result bitwise.
#   convert_b2e.sh DCP_STEP_DIR OUT_DIR
#     DCP_STEP_DIR  OLMo-core step dir with config.json and model_and_optim/ (fp32 masters; optimizer state unused)
#     OUT_DIR       new local dir: OUT_DIR/0/items (weights only, scan_layers=true, plain Orbax: no OCDBT, zarr v2),
#                   OUT_DIR/seed_receipt.json (sha256 of every seeded row before and after), OUT_DIR/bitwise.json
# Steps, as for the reference run (ledger H271):
#   1 convert_b2e_to_maxtext.py: the pinned converter (--mode olmo_core_to_maxtext, validate=true: its own round trip
#     must be 0.0) with rows 100256 / 100266 set to the fp32 means of [14023, 771, 29] / [524, 27963, 29] in the
#     embedding and the untied LM head, exactly as OLMo-core's PromotedTokenRowsCallback seeds them before SFT step 1.
#     The trainer must then not seed again (olmo_seed_token_rows=[]).
#   2 verify_b2e_bitwise.py: the items mapped back to OLMo-core keys equal the DCP's fp32 weights + seeding, bitwise
#     (reference: 829 tensors, 12.92B values).
# Then upload it yourself (printed at the end); nothing here writes to GCS or overwrites anything.
# Env:
#   PY           Python with MaxText (jason718/maxtext at a commit of ../image-olmo35-mm/maxtext_source), OLMo-core
#                fd9bba1ba, torch, jax[cpu] and tensorstore; the training image has all of them. Default python3.
#   MAXTEXT_SRC  optional MaxText src/ dir put first on PYTHONPATH (a checkout at the pin instead of an install).
#   DRY_RUN=1    print the commands only.
# Needs a CPU box with a few hundred GB of RAM: it holds several fp32 copies of the 12.9B values (~52 GB each).
set -euo pipefail
DCP=${1:?usage: convert_b2e.sh DCP_STEP_DIR OUT_DIR}
OUT=${2:?usage: convert_b2e.sh DCP_STEP_DIR OUT_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
DRY=${DRY_RUN:-}
if [[ -e "$OUT" ]]; then
  echo "$OUT exists; choose a new output dir (nothing is overwritten)" >&2
  exit 2
fi
if [[ -z "$DRY" && ! -f "$DCP/config.json" ]]; then
  echo "$DCP has no config.json; is it an OLMo-core step dir?" >&2
  exit 2
fi
export JAX_PLATFORMS=cpu
export PYTHONPATH="$HERE${MAXTEXT_SRC:+:$MAXTEXT_SRC}${PYTHONPATH:+:$PYTHONPATH}"
run() {
  echo "+ $*"
  [[ -n "$DRY" ]] || "$@"
}

run mkdir -p "$(dirname "$OUT")"
run "$PY" "$HERE/convert_b2e_to_maxtext.py" --receipt "$OUT/seed_receipt.json" \
  --mode olmo_core_to_maxtext --checkpoint_path "$DCP" --output_path "$OUT" \
  --scan_layers true --use_ocdbt false --use_zarr3 false --validate true --save_overwrite false
echo "+ $PY $HERE/verify_b2e_bitwise.py $DCP $OUT/0/items --out $OUT/bitwise.json"
if [[ -z "$DRY" ]]; then
  "$PY" "$HERE/verify_b2e_bitwise.py" "$DCP" "$OUT/0/items" --out "$OUT/bitwise.json" | tee "$OUT/bitwise.log"
  grep -q '^BITWISE_OK$' "$OUT/bitwise.log" || { echo "bitwise check failed, see $OUT/bitwise.json" >&2; exit 1; }
fi
echo "converted and verified: $OUT/0/items"
echo "upload to a new path, then point LOAD_PARAMETERS_PATH at <dest>/0/items:"
echo "  gcloud storage cp -r $OUT gs://ai2-olmo-dev/post-training/checkpoints/<new name>"
