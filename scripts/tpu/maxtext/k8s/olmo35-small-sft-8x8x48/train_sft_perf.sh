#!/bin/bash
# Olmo 3.5 Small text SFT in MaxText on a gke-tpu-v4 slice. Runs on every TPU host from the launcher ConfigMap
# (this script, perf_wrapper.py, maxtext_source); settings come from the env ConfigMap. The olmo-post-training-ledger's
# experiments/h269/h262-3k/train_sft_perf.sh, except that MaxText is the fork commit maxtext_source pins, with no
# patch step, and that template placeholders and an empty LOAD_PARAMETERS_PATH (random init) are refused.
set -euo pipefail
set -f # EXTRA_ARGS is split on spaces below; keep values like logical_axis_rules=[[...]] from being globbed.

: "${RUN_NAME:?}" "${OUTPUT_DIR:?}" "${LOAD_PARAMETERS_PATH:?}" "${TOKENIZER_PATH:?}" "${TRAIN_FILES:?}"
: "${TOTAL_DEVICES:?}" "${GLOBAL_BATCH:?}" "${STEPS:?}"
for v in RUN_NAME LOAD_PARAMETERS_PATH TRAIN_FILES STEPS LR_SCHEDULE_STEPS; do
  if [[ "${!v-}" == *CHANGE_ME* ]]; then
    echo "Set $v in your overlay before launching" >&2
    exit 2
  fi
done
MODEL_NAME="${MODEL_NAME:-olmo3-7b}"
SEQ_LEN="${SEQ_LEN:-4096}"
LR="${LR:-8e-5}"
WARMUP_FRAC="${WARMUP_FRAC:-0.03}"
DATA_SEED="${DATA_SEED:-0}"
NUM_EPOCH="${NUM_EPOCH:-2}" # Passes over the data the stream holds; STEPS decides how much is trained.
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
LR_SCHEDULE_STEPS="${LR_SCHEDULE_STEPS:-$STEPS}" # Schedule horizon; set it above STEPS to run the start of a longer recipe.
EXTRA_ARGS="${EXTRA_ARGS:-}"
export MAXTEXT_SFT_LABELS="${MAXTEXT_SFT_LABELS:-generation_blocks}" # or prefix (the SFT port on the fork, maxtext_source)

# PER_DEVICE_BATCH may be fractional when tensor or context parallelism shares one sequence across chips
# (e.g. 0.25 with ici_tensor_parallelism=4). Gradient accumulation makes up the rest of the global batch.
GRAD_ACCUM=$(python3 -c "
slots = $TOTAL_DEVICES * $PER_DEVICE_BATCH
assert slots == int(slots) and $GLOBAL_BATCH % int(slots) == 0, f'GLOBAL_BATCH=$GLOBAL_BATCH not divisible by TOTAL_DEVICES x PER_DEVICE_BATCH = {slots}'
print($GLOBAL_BATCH // int(slots))")

# The image (maxtext-posttrain, scripts/tpu/maxtext/image) carries the MaxText patches; fail fast if it doesn't.
python3 -c "from maxtext.input_pipeline import input_pipeline_utils as u; assert hasattr(u, '_split_conversation_into_segments')"
models_dir=$(python3 -c "import os, maxtext.models; print(os.path.dirname(maxtext.models.__file__))")

cat <<INFO
=== Olmo 3 SFT (MaxText) ===
  run          : $OUTPUT_DIR/$RUN_NAME
  init weights : $LOAD_PARAMETERS_PATH
  data         : $TRAIN_FILES (seed $DATA_SEED)
  tokenizer    : $TOKENIZER_PATH
  batch        : $TOTAL_DEVICES devices x $PER_DEVICE_BATCH x accum $GRAD_ACCUM = $GLOBAL_BATCH x $SEQ_LEN tokens/step
  model        : $MODEL_NAME, labels $MAXTEXT_SFT_LABELS
  steps        : $STEPS of a $LR_SCHEDULE_STEPS-step schedule, LR $LR, warmup $WARMUP_FRAC
  olmo3.py     : $(sha256sum "$models_dir/olmo3.py" | cut -c1-12)
  extra args   : ${EXTRA_ARGS:-<none>}
  LIBTPU_INIT_ARGS : ${LIBTPU_INIT_ARGS:-<none>}
INFO

export PYTHONUNBUFFERED=1
log=/tmp/train.log
# MaxText source: the fork commit in maxtext_source ("<https://github.com/owner/repo.git> <40-hex commit>"), as
# GitHub's tarball of that commit. Our MaxText changes are commits on the fork, so a change is a fork commit plus a new
# pin there. The image supplies the dependencies; that source and the Tunix it pins go first on PYTHONPATH.
read -r MAXTEXT_REPO SHA _ < "$(dirname "${BASH_SOURCE[0]}")/maxtext_source" || true
SLUG=${MAXTEXT_REPO#https://github.com/}
SLUG=${SLUG%.git}
if [[ ! "$SHA" =~ ^[0-9a-f]{40}$ || ! "$SLUG" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ || "$MAXTEXT_REPO" != "https://github.com/$SLUG.git" ]]; then
  echo "maxtext_source must hold '<https://github.com/owner/repo.git> <40-hex commit>'" >&2
  exit 2
fi
python3 - <<PY
import io, tarfile, urllib.request
data = urllib.request.urlopen("https://codeload.github.com/$SLUG/tar.gz/$SHA", timeout=300).read()
tarfile.open(fileobj=io.BytesIO(data)).extractall("/tmp")
PY
MT=/tmp/${SLUG#*/}-$SHA
TUNIX=https://github.com/google/tunix/archive/c2ee3bd8892fa7ceb29c5c2fd95416597e1582e2.zip
# A MaxText bump that moves its own Tunix pin must move this one too.
if ! grep -qF "$TUNIX" "$MT/src/dependencies/extra_deps/post_train_github_deps.txt"; then
  echo "MaxText $SHA pins another Tunix than $TUNIX; update the launcher's pin" >&2
  exit 2
fi
python3 -m pip install --quiet --no-deps --target /tmp/tunix-new "$TUNIX"
export PYTHONPATH=/tmp/tunix-new:$MT/src
python3 -c "import maxtext, tunix; print('maxtext from', maxtext.__file__, '| tunix from', tunix.__file__)"
cd $MT
# shellcheck disable=SC2086 # EXTRA_ARGS is meant to split.
python3 /launcher/perf_wrapper.py \
  src/maxtext/configs/post_train/sft.yml \
  run_name="$RUN_NAME" base_output_directory="$OUTPUT_DIR" model_name="$MODEL_NAME" \
  ${LOAD_PARAMETERS_PATH:+load_parameters_path="$LOAD_PARAMETERS_PATH"} tokenizer_path="$TOKENIZER_PATH" \
  hf_path=parquet hf_train_files="$TRAIN_FILES" train_split=train hf_eval_split="" eval_interval=-1 \
  train_data_columns="['messages']" sft_train_on_completion_only=true \
  max_target_length="$SEQ_LEN" packing=true num_epoch="$NUM_EPOCH" data_shuffle_seed="$DATA_SEED" \
  per_device_batch_size="$PER_DEVICE_BATCH" gradient_accumulation_steps="$GRAD_ACCUM" steps="$STEPS" \
  learning_rate_schedule_steps="$LR_SCHEDULE_STEPS" \
  learning_rate="$LR" lr_schedule_type=wsd warmup_steps_fraction="$WARMUP_FRAC" \
  wsd_decay_steps_fraction="$(python3 -c "print(1 - $WARMUP_FRAC)")" wsd_decay_style=linear learning_rate_final_fraction=0.0 \
  opt_type="${OPT_TYPE:-adamw}" adam_b1=0.9 adam_b2=0.95 adam_eps=1e-8 adam_weight_decay=0.0 gradient_clipping_threshold="${CLIP_THRESHOLD:-1.0}" \
  dtype=bfloat16 weight_dtype=float32 enable_dropout=false z_loss_multiplier=0.0 \
  $EXTRA_ARGS 2>&1 | tee "$log"
# train_sft exits 0 with zero steps when the data iterator fails (AI-Hypercomputer/maxtext#5523).
if ! grep -q "completed step: $STEPS," "$log"; then
  echo "STEP-GUARD: step $STEPS never completed" >&2
  exit 3
fi
echo "STEP-GUARD: reached step $STEPS"
