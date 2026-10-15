#!/bin/bash
# Olmo 3 DPO in MaxText on a gke-tpu-v4 slice (allenai/open-instruct#1933). Runs on every TPU host from the launcher
# ConfigMap; settings come from the env ConfigMap.
# Hyperparameters default to open-instruct's Olmo 3 7B Instruct DPO (scripts/train/olmo3/7b_instruct_dpo.sh with
# dpo.py's defaults): dpo_norm, beta 5, LR 1e-6 with 10% linear warmup then linear decay to 0, AdamW (0.9, 0.999)
# eps 1e-8, no weight decay, no gradient clipping, 128 pairs per step. Data is open-instruct's own tokenization
# (data/build_dpo_pretokenized.py), so tokens and loss masks match it by construction.
set -euo pipefail
set -f # EXTRA_ARGS is split on spaces below; keep values like logical_axis_rules=[[...]] from being globbed.

: "${RUN_NAME:?}" "${OUTPUT_DIR:?}" "${LOAD_PARAMETERS_PATH:?}" "${TOKENIZER_PATH:?}" "${TRAIN_FILES:?}"
: "${TOTAL_DEVICES:?}" "${GLOBAL_BATCH:?}" "${STEPS:?}"
MODEL_NAME="${MODEL_NAME:-olmo3-7b}"
SEQ_LEN="${SEQ_LEN:-16384}"
LR="${LR:-1e-6}"
WARMUP_FRAC="${WARMUP_FRAC:-0.1}"
BETA="${BETA:-5.0}"
LOSS_TYPE="${LOSS_TYPE:-dpo_norm}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
LR_SCHEDULE_STEPS="${LR_SCHEDULE_STEPS:-$STEPS}" # Schedule horizon; set it above STEPS to run the start of a longer recipe.
EXTRA_ARGS="${EXTRA_ARGS:-}"

# PER_DEVICE_BATCH counts pairs and may be fractional when tensor parallelism shares one pair across chips.
GRAD_ACCUM=$(python3 -c "
slots = $TOTAL_DEVICES * $PER_DEVICE_BATCH
assert slots == int(slots) and $GLOBAL_BATCH % int(slots) == 0, f'GLOBAL_BATCH=$GLOBAL_BATCH not divisible by TOTAL_DEVICES x PER_DEVICE_BATCH = {slots}'
print($GLOBAL_BATCH // int(slots))")

# The image (maxtext-posttrain, scripts/tpu/maxtext/image) carries the MaxText patches; fail fast if it doesn't.
python3 -c "from maxtext.input_pipeline import dpo_utils as u; assert hasattr(u, 'DPOPretokenizedFormatting')"

cat <<INFO
=== Olmo 3 DPO (MaxText) ===
  run          : $OUTPUT_DIR/$RUN_NAME
  init weights : $LOAD_PARAMETERS_PATH (also the frozen reference)
  data         : $TRAIN_FILES (pre-tokenized, file order)
  tokenizer    : $TOKENIZER_PATH
  batch        : $TOTAL_DEVICES devices x $PER_DEVICE_BATCH x accum $GRAD_ACCUM = $GLOBAL_BATCH pairs, each sequence padded to $SEQ_LEN
  loss         : $LOSS_TYPE, beta $BETA
  steps        : $STEPS of a $LR_SCHEDULE_STEPS-step schedule, LR $LR, warmup $WARMUP_FRAC
  extra args   : ${EXTRA_ARGS:-<none>}
  LIBTPU_INIT_ARGS : ${LIBTPU_INIT_ARGS:-<none>}
INFO

export PYTHONUNBUFFERED=1
log=/tmp/train.log
cd /deps
# num_epoch=2 only keeps a host from running dry before the last step; STEPS decides how much is trained.
# shellcheck disable=SC2086 # EXTRA_ARGS is meant to split.
python3 -m maxtext.trainers.post_train.dpo.train_dpo src/maxtext/configs/post_train/dpo.yml \
  run_name="$RUN_NAME" base_output_directory="$OUTPUT_DIR" model_name="$MODEL_NAME" \
  load_parameters_path="$LOAD_PARAMETERS_PATH" tokenizer_path="$TOKENIZER_PATH" \
  hf_path=parquet hf_train_files="$TRAIN_FILES" train_split=train hf_eval_split="" eval_interval=-1 \
  tokenize_train_data=false dpo.pretokenized=true dpo.loss_type="$LOSS_TYPE" dpo.dpo_beta="$BETA" \
  dpo.dpo_label_smoothing=0.0 \
  train_data_columns="['chosen_input_ids','chosen_labels','rejected_input_ids','rejected_labels']" \
  max_target_length="$SEQ_LEN" packing=false num_epoch=2 enable_data_shuffling=false \
  per_device_batch_size="$PER_DEVICE_BATCH" gradient_accumulation_steps="$GRAD_ACCUM" steps="$STEPS" \
  learning_rate_schedule_steps="$LR_SCHEDULE_STEPS" \
  learning_rate="$LR" lr_schedule_type=wsd warmup_steps_fraction="$WARMUP_FRAC" \
  wsd_decay_steps_fraction="$(python3 -c "print(1 - $WARMUP_FRAC)")" wsd_decay_style=linear learning_rate_final_fraction=0.0 \
  opt_type=adamw adam_b1=0.9 adam_b2=0.999 adam_eps=1e-8 adam_weight_decay=0.0 gradient_clipping_threshold=0 \
  dtype=bfloat16 weight_dtype=float32 enable_dropout=false \
  $EXTRA_ARGS 2>&1 | tee "$log"
# train_dpo exits 0 with zero steps when the data iterator fails (AI-Hypercomputer/maxtext#5523).
if ! grep -qE "(completed step: |Train step )$STEPS[ ,]" "$log"; then
  echo "STEP-GUARD: step $STEPS never completed" >&2
  exit 3
fi
echo "STEP-GUARD: reached step $STEPS"
