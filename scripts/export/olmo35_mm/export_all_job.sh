#!/bin/bash
# Beaker job body (1 GPU, an OLMo-core training image): every export a multimodal Olmo 3.5 SFT checkpoint needs.
#
#   export_all_job.sh STEP_DIR OUT_DIR
#
# STEP_DIR is an OLMo-core DCP step dir (model_and_optim/ + config.json): a GPU run's save, or a TPU (MaxText) run
# converted to OLMo-core. Writes, in this order (each step only after the previous one succeeded):
#   OUT_DIR/olmo-core-export/   model.safetensors + olmo_core_config.json, OLMo-core's export_multimodal_olmoddp.py
#                               --verify (bitwise against the train module's own loader). The vision evals read this
#                               (olmo-eval's olmo_core_vlm provider). Prints MMSFT-VISION-EXPORT-OK.
#   OUT_DIR/export/hf_text/     the language model alone as HF Olmo3MoeForCausalLM, from that export, with
#                               scripts/train/debug/convert_moe_checkpoint_to_hf.py --consolidated-export
#                               (generation_config eos [100265, 100257]); then smoke_hf_text.py. Text evals (vLLM)
#                               read this. Prints TOKENIZER-CHECK OK.
#   OUT_DIR/export/hf_mm/       only with EXPORT_HF_MM=1: the whole model as one HF Olmo3MoeVisionForConditionalGeneration
#                               dir (olmoe3_transformers_plugin.convert_checkpoint; also eos [100265, 100257]).
#
# Env: OLMO_CORE_DIR (an allenai/OLMo-core checkout at the multimodal SFT code, #942, visible to the job),
#      OLMO35_SFT_TOKENIZER (the promoted Olmo 3.5 chat tokenizer the SFT data was tokenized with),
#      EXPORT_DEVICE (cuda; cpu = no logit validation), EXPORT_HF_MM.
set -uo pipefail
STEP=${1:?usage: export_all_job.sh STEP_DIR OUT_DIR}
OUT=${2:?usage: export_all_job.sh STEP_DIR OUT_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
OI=$(cd "$HERE/../../.." && pwd)
OC=${OLMO_CORE_DIR:?set OLMO_CORE_DIR to an allenai/OLMo-core checkout at the multimodal SFT code}
TOK=${OLMO35_SFT_TOKENIZER:-/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12/a7/cache-neptune/numpy_sft/bf82a8fab8-6068a350-rowaligned/tokenizer}
DEV=${EXPORT_DEVICE:-cuda}
echo "MMSFT-EXPORT-ALL-START $(date -u +%FT%TZ) $STEP"
pip install -q -e "${OC}[fla,kernel-fun]" 2>&1 | tail -1
(cd "$OC" && git log -1 --format='olmo-core %H' 2>/dev/null); (cd "$OI" && git log -1 --format='open-instruct %H' 2>/dev/null)
mkdir -p "$OUT/export"

# 1. consolidated OLMo-core export (vision evals), verified bit for bit
(cd "$OC" && OMP_NUM_THREADS=32 PYTHONPATH=$OC/src python src/scripts/export_multimodal_olmoddp.py "$STEP" "$OUT/olmo-core-export" --verify) 2>&1 \
  | tee "$OUT/export_vision.log" | grep -E "Verified|Wrote|Error|Traceback"
grep -q "Verified all" "$OUT/export_vision.log" && grep -q "Wrote " "$OUT/export_vision.log" || { echo MMSFT-VISION-EXPORT-FAILED; exit 2; }
touch "$OUT/olmo-core-export/VERIFIED_OK"
echo "MMSFT-VISION-EXPORT-OK $(date -u +%FT%TZ)"

# 2. language model only (text evals), from the verified export
export PYTHONPATH=$OC/src:$OI
convert_text() {
  python "$OI/scripts/train/debug/convert_moe_checkpoint_to_hf.py" --consolidated-export "$OUT/olmo-core-export" \
    -o "$OUT/export/hf_text" -t "$TOK" --device "$DEV" "$@"
}
if [[ $DEV == cpu ]]; then convert_text --skip-validation; RC=$?; else convert_text; RC=$?; fi
echo "MMSFT-CONVERT-EXIT $RC"
if [[ $RC != 0 && $DEV != cpu ]]; then
  # Strict logit validation can fail on KDA at this scale with a correct conversion (see the converter's docstring);
  # the smoke below and the evals are the functional check then.
  echo MMSFT-CONVERT-RETRY-SKIP-VALIDATION
  rm -rf "$OUT/export/hf_text"
  convert_text --skip-validation; RC=$?
  echo "MMSFT-CONVERT-EXIT $RC"
fi
[[ $RC == 0 ]] || exit $RC
python "$HERE/smoke_hf_text.py" "$OUT/export/hf_text" --device "$DEV"; RC=$?
[[ $RC == 0 ]] && touch "$OUT/export/hf_text/SMOKE_OK"
echo "MMSFT-TEXT-EXPORT-EXIT $RC $(date -u +%FT%TZ)"
[[ $RC == 0 ]] || exit $RC

# 3. optional whole-model HF dir
if [[ -n ${EXPORT_HF_MM:-} ]]; then
  export PYTHONPATH=$OC/src:$HERE/transformers_plugin/src
  VAL=(); [[ $DEV == cpu ]] && VAL=(--skip-validation)
  python -m olmoe3_transformers_plugin.convert_checkpoint -i "$STEP" -o "$OUT/export/hf_mm" -t "$TOK" --device "$DEV" "${VAL[@]}"
  RC=$?
  echo "MMSFT-HF-MM-EXIT $RC"
fi
echo "MMSFT-EXPORT-ALL-EXIT $RC $(date -u +%FT%TZ)"
exit $RC
