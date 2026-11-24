#!/bin/bash
# Beaker job body (1 GPU): image+question sanity generation through olmo-eval's olmo_core_vlm provider (the
# provider the vision suites use) on one or more OLMo-core multimodal exports, side by side.
#
#   sanity_job.sh EXPORT_DIR [REFERENCE_EXPORT_DIR ...]
#
# Env (paths as the job sees them): OLMO_CORE_DIR (allenai/OLMo-core at the multimodal SFT code, #942),
# OLMO_EVAL_JOB_DIR (an allenai/olmo-eval checkout with the olmo_core_vlm provider's olmo35 layout, branch
# jasonr/olmo35-mmsft-eval), OLMO35_TOKENIZER (allenai/dolma2-tokenizer-olmo35@8b97170, on Weka), SANITY_IMAGE.
set -uo pipefail
[[ $# -ge 1 ]] || { sed -n 2,10p "$0"; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
OC=${OLMO_CORE_DIR:?set OLMO_CORE_DIR}
[[ -z ${OLMO_CORE_SHA:-} || $(git -C "$OC" rev-parse HEAD) == "$OLMO_CORE_SHA" ]] || { echo "$OC is not at OLMO_CORE_SHA=$OLMO_CORE_SHA" >&2; exit 1; }
OE=${OLMO_EVAL_JOB_DIR:?set OLMO_EVAL_JOB_DIR}
TOK=${OLMO35_TOKENIZER:-/weka/oe-training-default/jasonr/mmsft-eval/tokenizers/dolma2-tokenizer-olmo35-8b97170}
IMG=${SANITY_IMAGE:-/weka/oe-training-default/jasonr/mmsft-eval/tpu/sanity/coco_153632.jpg}
echo "SANITY-START $(date -u +%FT%TZ)"; python --version
pip install -q -e "${OC}[fla,kernel-fun]" 2>&1 | tail -1
pip install -q "flash-linear-attention==0.5.2" "omegaconf>=2.4.0.dev10,<2.5.0" "rich>=13.7,<15" prometheus_client 2>&1 | tail -1
export PYTHONPATH=$OE/src:$OC/src OMP_NUM_THREADS=8
python "$HERE/sanity_vlm.py" "$IMG" "$TOK" "$@"
RC=$?; echo "SANITY-EXIT $RC $(date -u +%FT%TZ)"; exit $RC
