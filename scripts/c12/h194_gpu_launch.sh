#!/bin/bash
# Ledger H194 (c12r-long): S-C's C12r SFT at --max_seq_length 131072 on 1 x 8 B300, with the 64K-128K LC replay
# dir (POOL) and --loss_implementation fused_linear. Same structure as h161_gpu_launch.sh (b1r arm), on one node.
# Writes a raw Beaker spec and submits it; DRY_RUN=1 prints the spec instead.
#
#   S_R=<steps> REPLAY=<POOL dir> scripts/c12/h194_gpu_launch.sh <tmax|noag> <smoke|resume|train>
#
#   tmax   olmo35-small-mm-ckpt2-256k-tmax step22320; noag: olmo35-small-mm-ckpt2-256k-noagentic step21159
#          (H181's two fixed-B2 candidates; H194's base is H181's pick).
#   S_R    two passes at 8 instances per step (H194 R4', from the offline loader replay); --num_epochs 2.
#   smoke  G2: 30 steps, every rank dumps its step-1 batch and logs batch hashes; then h161_refce.py on GPU 0
#          (the HF step-1 CE through H181's seeded HF copy of the base). Timeout 2 h.
#   resume G3: the smoke's schedule with autoResume, batch hashes, an ephemeral checkpoint at step 10 and a hold after
#          step 19 for `beaker job preempt`; the requeued job resumes from step10 and replays steps 11-19.
#   train  the run: permanent checkpoint at the end of epoch 1, ephemeral every SAVE_EVERY (1000), keep 2.
#
# Env: IMAGE (default the H111/H182 image), PRIORITY (high), MIN_RUNTIME, TIMEOUT, SAVE_EVERY, AUTO_RESUME,
# MAX_RETRIES, RUN_SUFFIX, WAIT_FOR_BASE=<minutes>, DRY_RUN=1.
set -euo pipefail
BASE="${1:?tmax or noag}"
MODE="${2:?smoke, resume or train}"
SEED=34521
HERE="$(cd "$(dirname "$0")" && pwd)"

IMAGE="${IMAGE:-01M5KX4ZAN00BJ00ARS7FNG79T}"
B=/weka/olmo-3p5-post-checkpoints/bases
H=/weka/olmo-3p5-post-checkpoints/abhishekr/tpu-posttrain/hf
case "$BASE" in
    tmax) MODEL=$B/olmo35-small-mm-ckpt2-256k-tmax/step22320-lm-olmocore/model_and_optim
          HF_REF=$H/olmo35-small-mm-ckpt2-256k-tmax-step22320-base-h149-c12seed ;;
    noag) MODEL=$B/olmo35-small-mm-ckpt2-256k-noagentic/step21159-lm-olmocore/model_and_optim
          HF_REF=$H/olmo35-small-mm-ckpt2-256k-noagentic-step21159-base-h149-c12seed ;;
    *) echo "unknown base $BASE" >&2; exit 1 ;;
esac
P=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12/data/parquets-a3
MIXER="$P/dolci-think-clean-v1-en-dbv-5bd68837-c12a3-9707a865.parquet 1.0 \
$P/nemotron-tool-oi-en-bd-825bdb83-c12a3-be9ec34a.parquet 0.862426 \
$P/nemotron-sci-v2-h083sel-bd-16f9aaaa-c12a3-25035bd6.parquet 3.03277 \
/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12/if/nemotron-ifchat-v3-if-decontam-1fee8794-c12-4e2daf40.parquet 4.937456"
OUT_ROOT=/weka/olmo-3p5-post-checkpoints/abhishekr/c12r-long
ROOTC=$OUT_ROOT/cache
CACHE=bf82a8fab8-0a2e7a2c-rowaligned
FROZEN=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12/a7/cache-neptune/numpy_sft/bf82a8fab8-6068a350-rowaligned
REPLAY="${REPLAY:?REPLAY=<H194 POOL dir>}"
STEPS="${S_R:?S_R=<steps from H194 R4 prime>}"
CKPT=$((STEPS / 2)); HOURS=96
SAVE_EVERY="${SAVE_EVERY:-1000}"
RUN_NAME="c12rlong-${BASE}-s${SEED}${RUN_SUFFIX:-}"
case "$MODE" in
    smoke) STEPS=30; CKPT=1000000; SAVE_EVERY=1024; HOURS=2; RUN_NAME="${RUN_NAME}-smoke"
           MIN_RUNTIME="${MIN_RUNTIME:-10m}"; AUTO_RESUME="${AUTO_RESUME:-0}" ;;
    resume) STEPS=30; CKPT=1000000; SAVE_EVERY=10; HOLD_AT=19; HOURS=3; RUN_NAME="${RUN_NAME}-resume"
           MIN_RUNTIME="${MIN_RUNTIME:-5m}"; AUTO_RESUME="${AUTO_RESUME:-1}"; MAX_RETRIES="${MAX_RETRIES:-1}" ;;
    train) MIN_RUNTIME="${MIN_RUNTIME:-1h}"; AUTO_RESUME="${AUTO_RESUME:-1}" ;;
    *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac
TIMEOUT="${TIMEOUT:-${HOURS}h}"
OUTPUT_DIR="$OUT_ROOT/$RUN_NAME"
DUMP_DIR="$OUTPUT_DIR/step1"

ARGS="--model_name_or_path $MODEL --config_name scripts/train/debug/kda_hero_small_sft.json \
--tokenizer_name_or_path allenai/dolma2-tokenizer-olmo35 --tokenizer_revision 8b9717061fae09d5be814373d189919a62a9a00d \
--chat_template_name olmo35 --reserved_slot_tokens '<think>' '</think>' \
--row_aligned_parts --document_boundaries_from_metadata --max_seq_length 131072 \
--per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 2e-4 --warmup_ratio 0.03 \
--weight_decay 0.0 --max_grad_norm 1.0 --num_epochs 2 --max_train_steps $STEPS --attn_implementation flash_2 \
--no_compile_model --activation_checkpointing_mode budget --activation_memory_budget 1 \
--loss_implementation fused_linear \
--checkpointing_steps $CKPT --ephemeral_save_interval $SAVE_EVERY --keep_last_n_checkpoints 2 --dist_timeout_hours 2 \
--no_save_async --with_tracking --run_name $RUN_NAME --logging_steps 1 --mixer_list $MIXER \
--local_cache_dir $ROOTC --seed 33333 --data_loader_seed $SEED --output_dir $OUTPUT_DIR \
--extra_numpy_dirs $REPLAY"

BASE_RECEIPT="$(dirname "$MODEL")/RECEIPT.json"
WRAP=$(gzip -9c "$HERE/h194_wrap.py" | base64 | tr -d '\n')
SWRAP=$(gzip -9c "$HERE/h161_smoke_wrap.py" | base64 | tr -d '\n')
PRE="set -uo pipefail
printf %s '$WRAP' | base64 -d | gunzip > /tmp/h194_wrap.py
printf %s '$SWRAP' | base64 -d | gunzip > /tmp/h161_smoke_wrap.py
sha256sum /tmp/h194_wrap.py /tmp/h161_smoke_wrap.py
echo H194-JOB-START \$(date -u +%FT%TZ) job \$BEAKER_JOB_ID
ls -d $OUTPUT_DIR/step* 2>/dev/null
mkdir -p $ROOTC/numpy_sft
[ -e $ROOTC/numpy_sft/$CACHE ] || ln -s $FROZEN $ROOTC/numpy_sft/$CACHE
ls -l $ROOTC/numpy_sft/$CACHE
for i in \$(seq 1 ${WAIT_FOR_BASE:-0}); do [ -f $BASE_RECEIPT ] && break; sleep 60; done
[ -f $BASE_RECEIPT ] || { echo H194-NO-BASE $BASE_RECEIPT; exit 3; }
sha256sum $BASE_RECEIPT"
TORCHRUN="torchrun --nnodes=1 --nproc_per_node=8 /tmp/h194_wrap.py $ARGS"
EXTRA_ENV=""
if [[ "$MODE" == resume ]]; then
    CMD="$PRE
$TORCHRUN; RC=\$?
echo H194-TRAIN-EXIT \$RC \$(date -u +%FT%TZ)
exit \$RC"
    EXTRA_ENV='{"name": "H161_BATCH_HASH", "value": "1"}, {"name": "H161_HOLD_AT_STEP", "value": "'"$HOLD_AT"'"},
    {"name": "H161_NO_FINAL_SAVE", "value": "1"},'
elif [[ "$MODE" == smoke ]]; then
    REFCE=$(gzip -9c "$HERE/h161_refce.py" | base64 | tr -d '\n')
    # As h161_gpu_launch.sh's smoke: exit is training's code if it failed; else 4 if the reference failed, 5 if a
    # rank dump is missing.
    CMD="$PRE
printf %s '$REFCE' | base64 -d | gunzip > /tmp/h161_refce.py
sha256sum /tmp/h161_refce.py
$TORCHRUN; RC=\$?
echo H194-TRAIN-EXIT \$RC \$(date -u +%FT%TZ)
N=\$(ls $DUMP_DIR/rank*.npz 2>/dev/null | wc -l); echo H194-DUMPS \$N
if [ \"\$N\" = 8 ]; then
  command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl >/dev/null)
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; export PATH=\$HOME/.local/bin:\$PATH; }
  PX=''
  for c in /usr/local/cuda/bin/ptxas /usr/local/cuda-13*/bin/ptxas; do [ -x \"\$c\" ] && \"\$c\" --help 2>&1 | grep -q sm_103a && { PX=\$c; break; }; done
  if [ -z \"\$PX\" ]; then uv venv -q --python 3.12 /tmp/nvcc13 && uv pip install -q --python /tmp/nvcc13/bin/python 'nvidia-cuda-nvcc==13.0.*'; PX=\$(find /tmp/nvcc13 -name ptxas -type f | head -1); chmod +x \"\$PX\"; fi
  echo PTXAS \$PX; export TRITON_PTXAS_PATH=\$PX
  cd /tmp && CUDA_VISIBLE_DEVICES=0 uv run --no-project --python 3.12 --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match --with torch==2.9.1 --with transformers==4.57.6 --with numpy --with accelerate --with flash-linear-attention python /tmp/h161_refce.py $DUMP_DIR $HF_REF $DUMP_DIR/refce.json
  RR=\$?; echo H194-REFCE-EXIT \$RR
  [ \"\$RC\" = 0 ] && [ \"\$RR\" != 0 ] && RC=4
else
  [ \"\$RC\" = 0 ] && RC=5
fi
exit \$RC"
    EXTRA_ENV='{"name": "H161_DUMP_DIR", "value": "'"$DUMP_DIR"'"}, {"name": "H161_NO_FINAL_SAVE", "value": "1"},
    {"name": "H161_BATCH_HASH", "value": "1"},'
else
    CMD="$PRE
$TORCHRUN"
fi

SPEC=$(mktemp -t h194spec.XXXXXX)
CMD="$CMD" python3 - "$SPEC" <<EOF
import json, os, sys
env = [
    {"name": "OLMO_SHARED_FS", "value": "1"},
    {"name": "C12R_CACHE_ROOT", "value": "$ROOTC"},
    {"name": "EXPECTED_NUMPY_CACHE", "value": "$CACHE"},
    {"name": "HF_HOME", "value": "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c11/hf_cache/huggingface"},
    {"name": "HF_DATASETS_CACHE", "value": "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c11/hf_cache/huggingface"},
    {"name": "HF_HUB_CACHE", "value": "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c11/hf_cache/hub"},
    {"name": "NCCL_DEBUG", "value": "ERROR"},
    $EXTRA_ENV
    {"name": "HF_TOKEN", "secret": "abhishekr_HF_TOKEN"},
    {"name": "WANDB_API_KEY", "secret": "abhishekr_WANDB_API_KEY"},
    {"name": "BEAKER_TOKEN", "secret": "abhishekr_BEAKER_TOKEN"},
]
task = {
    "name": "$RUN_NAME",
    "image": {"beaker": "$IMAGE"},
    "command": ["/bin/bash", "-c"],
    "arguments": [os.environ["CMD"]],
    "envVars": env,
    "datasets": [{"mountPath": f"/weka/{b}", "source": {"weka": b}} for b in
                 ("oe-adapt-default", "oe-training-default", "olmo-3p5-checkpoints", "olmo-3p5-post-checkpoints")],
    "result": {"path": "/output"},
    "resources": {"gpuCount": 8, "sharedMemory": "10GiB"},
    "context": {"priority": "${PRIORITY:-high}", "minRuntime": "$MIN_RUNTIME", "autoResume": "$AUTO_RESUME" == "1"},
    "constraints": {"cluster": ["ai2/holmes"]},
    "timeout": "$TIMEOUT",
}
spec = {"version": "v2", "budget": "ai2/oe-other",
        "description": "ledger H194 c12r-long: $RUN_NAME, $STEPS steps, 1x8 B300, seq 131072, fused_linear, lr 2e-4",
        "tasks": [task]}
if int("${MAX_RETRIES:-0}"):
    spec["retry"] = {"allowedTaskRetries": int("${MAX_RETRIES:-0}")}
json.dump(spec, open(sys.argv[1], "w"), indent=1)
EOF
if [[ "${DRY_RUN:-0}" == 1 ]]; then cat "$SPEC"; exit 0; fi
beaker experiment create "$SPEC" --workspace ai2/olmo-instruct --name "$RUN_NAME-$(date +%m%d%H%M)"
