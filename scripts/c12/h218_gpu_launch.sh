#!/bin/bash
# Ledger H218 / H219 (redsearcher-noag / redsearcher-tmax): continued SFT of H181's C12r checkpoints on the converted
# REDSearcher set (allenai/redsearcher-sft-10k-browsecomp-tools@954cf9d8) at --max_seq_length 131072 on 1 x 8 B300,
# H194's configuration (fused_linear, row_aligned_parts, document_boundaries_from_metadata, olmo35 template, eager)
# with H208's continuation schedule: LR 2e-5, warmup 3 %, linear to 0 over two epochs. Same structure as
# h194_gpu_launch.sh, on one node. Writes a raw Beaker spec and submits it; DRY_RUN=1 prints the spec instead.
#
#   scripts/c12/h218_gpu_launch.sh <noag|tmax> <tokenize|smoke|train>
#
#   noag   start = H181 b2noag-c12r step 18,083 (H218); tmax = H181 b2tmax-c12r step 18,083 (H219). Each start is
#          the HF export written as an OLMo-core DCP by scripts/c12/h218_hf_to_olmocore_dcp.py (RECEIPT.json beside
#          it); the HF export itself is the step-1 CE reference.
#   tokenize  gate (1)+(2): a CPU job builds the row-aligned numpy cache under $ROOT/cache (the same cache serves
#          both starts: it depends on the data, tokenizer, template, length and seed, not the model), then
#          scripts/c12/h218_mask_dump.py writes the cache receipt and the label-mask dump of the 20 rows with the
#          most tool turns to $ROOT/receipts, re-tokenizing them with the start's own tokenizer. Prints the cache
#          name to set as EXPECTED_NUMPY_CACHE for smoke and train.
#   smoke  gate (3)+(4): SMOKE_STEPS (20) steps at the production size and LR; every rank dumps its step-1 batch and
#          logs batch hashes; then h161_refce.py on GPU 0 forwards the dumped batch through the HF export (step-1 CE
#          within 1e-3). Timeout 2 h. Needs EXPECTED_NUMPY_CACHE.
#   train  the run: S_R steps (= 2 x floor(I / 8) from the cache receipt, two epochs), permanent checkpoint half
#          way, ephemeral every SAVE_EVERY (200), keep 2, final save at S_R. Needs EXPECTED_NUMPY_CACHE and S_R.
#
# Env: IMAGE (H194's), PRIORITY, MIN_RUNTIME, TIMEOUT, SAVE_EVERY, AUTO_RESUME, MAX_RETRIES, RUN_SUFFIX, SMOKE_STEPS,
# EXPECTED_NUMPY_CACHE, S_R, DRY_RUN=1.
set -euo pipefail
BASE="${1:?noag or tmax}"
MODE="${2:?tokenize, smoke or train}"
SEED=34521
HERE="$(cd "$(dirname "$0")" && pwd)"

IMAGE="${IMAGE:-01M5KX4ZAN00BJ00ARS7FNG79T}"
ROOT=/weka/olmo-3p5-post-checkpoints/pradeepd/h218
H=/weka/olmo-3p5-post-checkpoints/abhishekr/tpu-posttrain/hf
case "$BASE" in
    noag|tmax) HF_REF=$H/olmo35-sft-h181-b2${BASE}-c12r-step18083
               MODEL=$ROOT/starts/olmo35-sft-h181-b2${BASE}-c12r-step18083-olmocore/model_and_optim ;;
    *) echo "unknown base $BASE" >&2; exit 1 ;;
esac
DATA=/weka/oe-adapt-default/pradeepd/h218/data/redsearcher-sft-10k-browsecomp-tools-954cf9d8.parquet
MIXER="$DATA 1.0"
ROOTC=$ROOT/cache
SAVE_EVERY="${SAVE_EVERY:-200}"
RUN_NAME="rs-${BASE}-s${SEED}${RUN_SUFFIX:-}"
GPUS=8
case "$MODE" in
    tokenize) STEPS=1; CKPT=1000000; HOURS=4; RUN_NAME="${RUN_NAME}-tokenize"; GPUS=0
           MIN_RUNTIME="${MIN_RUNTIME:-0s}"; AUTO_RESUME="${AUTO_RESUME:-0}" ;;
    smoke) STEPS="${SMOKE_STEPS:-20}"; CKPT=1000000; SAVE_EVERY=1024; HOURS=2; RUN_NAME="${RUN_NAME}-smoke"
           MIN_RUNTIME="${MIN_RUNTIME:-0s}"; AUTO_RESUME="${AUTO_RESUME:-0}" ;;
    train) STEPS="${S_R:?S_R=<2 x floor(I / 8) from the cache receipt>}"; CKPT=$(((STEPS + 1) / 2)); HOURS=24
           MIN_RUNTIME="${MIN_RUNTIME:-8h}"; AUTO_RESUME="${AUTO_RESUME:-1}" ;;
    *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac
if [[ "$MODE" != tokenize ]]; then : "${EXPECTED_NUMPY_CACHE:?EXPECTED_NUMPY_CACHE=<cache name from the tokenize job>}"; fi
TIMEOUT="${TIMEOUT:-${HOURS}h}"
OUTPUT_DIR="$ROOT/runs/$RUN_NAME"
DUMP_DIR="$OUTPUT_DIR/step1"

ARGS="--model_name_or_path $MODEL --config_name scripts/train/debug/kda_hero_small_sft.json \
--tokenizer_name_or_path allenai/dolma2-tokenizer-olmo35 --tokenizer_revision 8b9717061fae09d5be814373d189919a62a9a00d \
--chat_template_name olmo35 --reserved_slot_tokens '<think>' '</think>' \
--row_aligned_parts --document_boundaries_from_metadata --max_seq_length 131072 \
--per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 2e-5 --warmup_ratio 0.03 \
--weight_decay 0.0 --max_grad_norm 1.0 --num_epochs 2 --max_train_steps $STEPS --attn_implementation flash_2 \
--no_compile_model --activation_checkpointing_mode budget --activation_memory_budget 1 \
--loss_implementation fused_linear \
--checkpointing_steps $CKPT --ephemeral_save_interval $SAVE_EVERY --keep_last_n_checkpoints 2 --dist_timeout_hours 2 \
--no_save_async --with_tracking --run_name $RUN_NAME --logging_steps 1 --mixer_list $MIXER \
--local_cache_dir $ROOTC --seed 33333 --data_loader_seed $SEED --output_dir $OUTPUT_DIR"

WRAP=$(gzip -9c "$HERE/h218_wrap.py" | base64 | tr -d '\n')
SWRAP=$(gzip -9c "$HERE/h161_smoke_wrap.py" | base64 | tr -d '\n')
MASK=$(gzip -9c "$HERE/h218_mask_dump.py" | base64 | tr -d '\n')
PRE="set -uo pipefail
printf %s '$WRAP' | base64 -d | gunzip > /tmp/h218_wrap.py
printf %s '$SWRAP' | base64 -d | gunzip > /tmp/h161_smoke_wrap.py
printf %s '$MASK' | base64 -d | gunzip > /tmp/h218_mask_dump.py
sha256sum /tmp/h218_wrap.py /tmp/h161_smoke_wrap.py /tmp/h218_mask_dump.py
echo H218-JOB-START \$(date -u +%FT%TZ) job \$BEAKER_JOB_ID mode $MODE base $BASE
mkdir -p $ROOTC/numpy_sft $ROOT/receipts $ROOT/runs
sha256sum $DATA | tee $ROOT/receipts/data_sha256.txt"
TORCHRUN="torchrun --nnodes=1 --nproc_per_node=8 /tmp/h218_wrap.py $ARGS"
EXTRA_ENV=""
if [[ "$MODE" == tokenize ]]; then
    # No GPU: olmo_core_finetune.py returns from --cache_dataset_only before any distributed init. The cache name is
    # the config hash + seed/length suffix + -rowaligned; the receipt and the mask dump go to \$ROOT/receipts.
    CMD="$PRE
ls $ROOTC/numpy_sft/ || true
RANK=0 python /tmp/h218_wrap.py $ARGS --cache_dataset_only; RC=\$?
echo H218-TOKENIZE-EXIT \$RC \$(date -u +%FT%TZ)
[ \"\$RC\" = 0 ] || exit \$RC
CACHE=\$(ls -td $ROOTC/numpy_sft/*-rowaligned | head -1); echo H218-CACHE-DIR \$CACHE
ls -la \$CACHE | head -30; cat \$CACHE/dataset_statistics.json | head -c 4000; echo
python /tmp/h218_mask_dump.py \$CACHE $ROOT/receipts/mask_\$(basename \$CACHE).json --rows 20 --max-seq-length 131072 --check-tokenizer $HF_REF; RM=\$?
echo H218-MASK-EXIT \$RM
echo EXPECTED_NUMPY_CACHE=\$(basename \$CACHE)
exit \$RM"
elif [[ "$MODE" == smoke ]]; then
    REFCE=$(gzip -9c "$HERE/h161_refce.py" | base64 | tr -d '\n')
    # As h194_gpu_launch.sh's smoke: exit is training's code if it failed; else 4 if the reference failed, 5 if a
    # rank dump is missing.
    CMD="$PRE
printf %s '$REFCE' | base64 -d | gunzip > /tmp/h161_refce.py
sha256sum /tmp/h161_refce.py
sha256sum $(dirname $MODEL)/RECEIPT.json
$TORCHRUN; RC=\$?
echo H218-TRAIN-EXIT \$RC \$(date -u +%FT%TZ)
N=\$(ls $DUMP_DIR/rank*.npz 2>/dev/null | wc -l); echo H218-DUMPS \$N
if [ \"\$N\" = 8 ]; then
  command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl >/dev/null)
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; export PATH=\$HOME/.local/bin:\$PATH; }
  PX=''
  for c in /usr/local/cuda/bin/ptxas /usr/local/cuda-13*/bin/ptxas; do [ -x \"\$c\" ] && \"\$c\" --help 2>&1 | grep -q sm_103a && { PX=\$c; break; }; done
  if [ -z \"\$PX\" ]; then uv venv -q --python 3.12 /tmp/nvcc13 && uv pip install -q --python /tmp/nvcc13/bin/python 'nvidia-cuda-nvcc==13.0.*'; PX=\$(find /tmp/nvcc13 -name ptxas -type f | head -1); chmod +x \"\$PX\"; fi
  echo PTXAS \$PX; export TRITON_PTXAS_PATH=\$PX
  cd /tmp && CUDA_VISIBLE_DEVICES=0 uv run --no-project --python 3.12 --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match --with torch==2.9.1 --with transformers==${REFCE_TRANSFORMERS:-4.57.6} --with numpy --with accelerate --with flash-linear-attention python /tmp/h161_refce.py $DUMP_DIR $HF_REF $DUMP_DIR/refce.json
  RR=\$?; echo H218-REFCE-EXIT \$RR
  [ \"\$RC\" = 0 ] && [ \"\$RR\" != 0 ] && RC=4
else
  [ \"\$RC\" = 0 ] && RC=5
fi
exit \$RC"
    EXTRA_ENV='{"name": "H161_DUMP_DIR", "value": "'"$DUMP_DIR"'"}, {"name": "H161_NO_FINAL_SAVE", "value": "1"},
    {"name": "H161_BATCH_HASH", "value": "1"},'
else
    CMD="$PRE
sha256sum $(dirname $MODEL)/RECEIPT.json
$TORCHRUN; RC=\$?
echo H218-TRAIN-EXIT \$RC \$(date -u +%FT%TZ)
exit \$RC"
fi

SPEC=$(mktemp -t h218spec.XXXXXX)
CMD="$CMD" python3 - "$SPEC" <<EOF
import json, os, sys
env = [
    {"name": "OLMO_SHARED_FS", "value": "1"},
    {"name": "H218_CACHE_ROOT", "value": "$ROOTC"},
    {"name": "H218_PROMOTED_IDS", "value": "100256,100266"},
    {"name": "EXPECTED_NUMPY_CACHE", "value": "${EXPECTED_NUMPY_CACHE:-}"},
    {"name": "HF_HOME", "value": "/weka/oe-adapt-default/pradeepd/h218/hf_cache"},
    {"name": "HF_DATASETS_CACHE", "value": "/weka/oe-adapt-default/pradeepd/h218/hf_cache"},
    {"name": "HF_HUB_CACHE", "value": "/weka/oe-adapt-default/pradeepd/h218/hf_cache/hub"},
    {"name": "NCCL_DEBUG", "value": "ERROR"},
    $EXTRA_ENV
    {"name": "HF_TOKEN", "secret": "pradeepd_HF_TOKEN"},
    {"name": "WANDB_API_KEY", "secret": "pradeepd_WANDB_API_KEY"},
    {"name": "BEAKER_TOKEN", "secret": "pradeepd_BEAKER_TOKEN"},
]
if not os.environ.get("EXPECTED_NUMPY_CACHE"):
    env = [e for e in env if e["name"] != "EXPECTED_NUMPY_CACHE"]
gpus = int("$GPUS")
task = {
    "name": "$RUN_NAME",
    "image": {"beaker": "$IMAGE"},
    "command": ["/bin/bash", "-c"],
    "arguments": [os.environ["CMD"]],
    "envVars": env,
    "datasets": [{"mountPath": f"/weka/{b}", "source": {"weka": b}} for b in
                 ("oe-adapt-default", "oe-training-default", "olmo-3p5-checkpoints", "olmo-3p5-post-checkpoints")],
    "result": {"path": "/output"},
    "resources": ({"gpuCount": gpus, "sharedMemory": "10GiB"} if gpus else {"cpuCount": 16, "memory": "96GiB"}),
    "context": {"priority": "${PRIORITY:-urgent}", "minRuntime": "$MIN_RUNTIME", "autoResume": "$AUTO_RESUME" == "1"},
    "constraints": {"cluster": (["ai2/holmes"] if gpus else ["ai2/jupiter", "ai2/holmes"])},
    "timeout": "$TIMEOUT",
}
spec = {"version": "v2", "budget": "ai2/oe-other",
        "description": "ledger H218/H219 redsearcher continuation: $RUN_NAME, $MODE, $STEPS steps, 1x8 B300, seq 131072, fused_linear, lr 2e-5, two epochs",
        "tasks": [task]}
if int("${MAX_RETRIES:-0}"):
    spec["retry"] = {"allowedTaskRetries": int("${MAX_RETRIES:-0}")}
json.dump(spec, open(sys.argv[1], "w"), indent=1)
EOF
if [[ "${DRY_RUN:-0}" == 1 ]]; then cat "$SPEC"; exit 0; fi
beaker experiment create "$SPEC" --workspace ai2/olmo-instruct --name "$RUN_NAME-$(date +%m%d%H%M)"
