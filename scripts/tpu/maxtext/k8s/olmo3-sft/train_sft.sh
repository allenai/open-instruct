#!/bin/bash
# Olmo 3 SFT in MaxText on a gke-tpu-v4 slice (allenai/open-instruct#1933). Runs on every TPU host from the
# olmo3-sft-launcher ConfigMap; settings come from the olmo3-sft-env ConfigMap.
# Hyperparameters mirror the GPU parity runs (open-instruct OLMo-core SFT at 118262550): 64 x 4096 packed
# tokens per step, LR 8e-5 linear decay with 3% warmup, AdamW (0.9, 0.95), no weight decay, clip 1.0.
set -euo pipefail
set -f # EXTRA_ARGS is split on spaces below; keep values like logical_axis_rules=[[...]] from being globbed.

: "${RUN_NAME:?}" "${OUTPUT_DIR:?}" "${LOAD_PARAMETERS_PATH:?}" "${TOKENIZER_PATH:?}" "${TRAIN_FILES:?}"
: "${TOTAL_DEVICES:?}" "${GLOBAL_BATCH:?}" "${STEPS:?}"
SEQ_LEN="${SEQ_LEN:-4096}"
LR="${LR:-8e-5}"
WARMUP_FRAC="${WARMUP_FRAC:-0.03}"
DATA_SEED="${DATA_SEED:-0}"
NUM_EPOCH="${NUM_EPOCH:-2}" # Passes over the data the stream holds; STEPS decides how much is trained.
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

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
  steps        : $STEPS, LR $LR, warmup $WARMUP_FRAC
  olmo3.py     : $(sha256sum "$models_dir/olmo3.py" | cut -c1-12)
  extra args   : ${EXTRA_ARGS:-<none>}
  LIBTPU_INIT_ARGS : ${LIBTPU_INIT_ARGS:-<none>}
INFO

export PYTHONUNBUFFERED=1
log=/tmp/train.log
cd /deps
# shellcheck disable=SC2086 # EXTRA_ARGS is meant to split.
python3 -m maxtext.trainers.post_train.sft.train_sft src/maxtext/configs/post_train/sft.yml \
  run_name="$RUN_NAME" base_output_directory="$OUTPUT_DIR" model_name=olmo3-7b \
  load_parameters_path="$LOAD_PARAMETERS_PATH" tokenizer_path="$TOKENIZER_PATH" \
  hf_path=parquet hf_train_files="$TRAIN_FILES" train_split=train hf_eval_split="" eval_interval=-1 \
  train_data_columns="['messages']" sft_train_on_completion_only=true \
  max_target_length="$SEQ_LEN" packing=true num_epoch="$NUM_EPOCH" data_shuffle_seed="$DATA_SEED" \
  per_device_batch_size="$PER_DEVICE_BATCH" gradient_accumulation_steps="$GRAD_ACCUM" steps="$STEPS" \
  learning_rate="$LR" lr_schedule_type=wsd warmup_steps_fraction="$WARMUP_FRAC" \
  wsd_decay_steps_fraction="$(python3 -c "print(1 - $WARMUP_FRAC)")" wsd_decay_style=linear learning_rate_final_fraction=0.0 \
  opt_type=adamw adam_b1=0.9 adam_b2=0.95 adam_eps=1e-8 adam_weight_decay=0.0 gradient_clipping_threshold=1.0 \
  dtype=bfloat16 weight_dtype=float32 enable_dropout=false z_loss_multiplier=0.0 \
  $EXTRA_ARGS 2>&1 | tee "$log"
# train_sft exits 0 with zero steps when the data iterator fails (AI-Hypercomputer/maxtext#5523).
if ! grep -q "completed step: $STEPS," "$log"; then
  echo "STEP-GUARD: step $STEPS never completed" >&2
  exit 3
fi
echo "STEP-GUARD: reached step $STEPS"
