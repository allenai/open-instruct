#!/bin/bash
# Beaker job body (1 GPU): EMo HF-vs-native logit check of an export made by export_all_job.sh.
#
#   emo_check_job.sh STEP_DIR OUT_DIR
#
# STEP_DIR is the DCP step dir the export came from (its config.json is the training config, with the EMo router
# settings the export drops); OUT_DIR holds olmo-core-export/ and export/hf_text/. Prints EMO-LOGIT-CHECK
# PASS|FAIL|SKIP (SKIP: the run has no EMo router). Env: OLMO_CORE_DIR as export_all_job.sh.
set -uo pipefail
STEP=${1:?usage: emo_check_job.sh STEP_DIR OUT_DIR}
OUT=${2:?usage: emo_check_job.sh STEP_DIR OUT_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
OC=${OLMO_CORE_DIR:?set OLMO_CORE_DIR to an allenai/OLMo-core checkout at the multimodal SFT code}
[[ -z ${OLMO_CORE_SHA:-} || $(git -C "$OC" rev-parse HEAD) == "$OLMO_CORE_SHA" ]] || { echo "$OC is not at OLMO_CORE_SHA=$OLMO_CORE_SHA" >&2; exit 1; }
pip install -q -e "${OC}[fla,kernel-fun]" 2>&1 | tail -1
export PYTHONPATH=$OC/src:$HERE/transformers_plugin/src
CFG=$STEP/config.json
[[ -f $CFG ]] || { echo "no training config at $CFG"; exit 2; }
echo "training config: $CFG"; cp "$CFG" "$OUT/training_config.json"
python "$HERE/emo_logit_check.py" "$OUT/olmo-core-export" "$OUT/export/hf_text" "$OUT/training_config.json" 2>&1 \
  | tee "$OUT/emo_logit_check.log"
exit "${PIPESTATUS[0]}"
