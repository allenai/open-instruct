#!/bin/bash
# Olmo 3.5 multimodal SFT (OLMo-core's olmo35_mm_sft at fd9bba1ba, the GPU arm H231 emo256k) in MaxText. Runs on every
# TPU host of the olmo35-mm-sft JobSet from its -launcher ConfigMap; settings come from its -env ConfigMap.
# From allenai/olmo-gcp-infra@3575547e82325258583ff7f83e3e7e123d1e7826 examples/olmo35-maxtext/k8s/mm-sft-4x4x8/train.sh
# (the reference run's launcher), plus the RUN_NAME guard of ../olmo3-sft/train_sft.sh and DRY_RUN=1. The recipe lives
# in CONFIG and TRAIN_ARGS, the slice layout in SHARD_ARGS, everything else in EXTRA_ARGS.
#
# On exit, success or not, every host copies /tmp/tpu_logs/tpu_driver.ERROR (and the tail of its own log) to
# OUTPUT_DIR/RUN_NAME/host-logs/. A run that exits 0 without completing a single step fails (STEP-GUARD).
# DRY_RUN=1 prints the settings and the train command, then exits without touching TPUs, GCS or pip.
set -euo pipefail

: "${RUN_NAME:?}"   # Checkpoints and logs go under OUTPUT_DIR/RUN_NAME. Keep it to resume.
: "${OUTPUT_DIR:?}" # MaxText base_output_directory, gs://...
: "${STEPS:?}"
: "${LOAD_PARAMETERS_PATH:?}" # Orbax items dir of the converted base (B2e step 22240 + <think> rows).
if [[ "$RUN_NAME" == CHANGE_ME ]]; then
  echo "Set RUN_NAME to a unique name in your overlay before launching" >&2
  exit 2
fi
DRY_RUN="${DRY_RUN:-}"
CONFIG="${CONFIG:-/deps/src/maxtext/configs/olmo/olmo35_tiny_mm_sft_64k.yml}"
CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-500}"
OLMO_MM_CONFIG="${OLMO_MM_CONFIG:-}"   # Optional: a GPU run's config.json for dataset_type=olmo_mm (olmo_mm_config_path).
OLMO_MM_WORKERS="${OLMO_MM_WORKERS:-}" # Optional: olmo_mm_num_workers.
SHARD_ARGS="${SHARD_ARGS:-}"
TRAIN_ARGS="${TRAIN_ARGS:-}" # The recipe's command-line overrides (kustomization.yaml explains each).
EXTRA_ARGS="${EXTRA_ARGS:-}" # Last, so they override everything above.
WANDB_SECRET="${WANDB_SECRET:-}"
WANDB_PROJECT="${WANDB_PROJECT:-olmo35-maxtext}"
WANDB_VERSION="${WANDB_VERSION:-0.30.0}"
HOST="${JOB_COMPLETION_INDEX:-$(hostname)}"
LOG=/tmp/train.log

if [[ -z "$DRY_RUN" && ! -f "$CONFIG" ]]; then
  echo "ERROR: $CONFIG is not in the image. Was it built from the MaxText branch with the multimodal SFT config?" >&2
  exit 1
fi
if [[ -z "$DRY_RUN" && -n "$OLMO_MM_CONFIG" && "$OLMO_MM_CONFIG" != gs://* && ! -f "$OLMO_MM_CONFIG" ]]; then
  echo "ERROR: $OLMO_MM_CONFIG not found. Is the gs://midtraining-data/weka subfolder holding it mounted?" >&2
  exit 1
fi

# Copy the libtpu error log and the tail of this host's log to GCS whatever happens, so a crash that takes the pod
# with it still leaves its reason behind.
upload_logs() {
  local rc=$?
  set +e
  python3 - "$OUTPUT_DIR/$RUN_NAME/host-logs" "$HOST" "$LOG" <<'PY' || true
import os, subprocess, sys
from google.cloud import storage
dest, host, log = sys.argv[1:]
bucket_name, _, prefix = dest[len("gs://"):].partition("/")
bucket = storage.Client().bucket(bucket_name)
for src, name in (("/tmp/tpu_logs/tpu_driver.ERROR", "tpu_driver.ERROR"), (log, "train.log.tail")):
    if not os.path.isfile(src):
        continue
    data = open(src, "rb").read()
    if name.endswith(".tail"):
        data = data[-2_000_000:]
    blob = bucket.blob(f"{prefix}/host-{host}-{name}")
    blob.upload_from_string(data)
    print(f"uploaded {src} -> gs://{bucket_name}/{blob.name}")
PY
  exit "$rc"
}
[[ -n "$DRY_RUN" ]] || trap upload_logs EXIT

data_args=()
if [[ -n "$OLMO_MM_CONFIG" ]]; then
  data_args+=(dataset_type=olmo_mm olmo_mm_config_path="$OLMO_MM_CONFIG" olmo_mm_work_dir=/tmp/olmo_mm_work)
fi
if [[ -n "$OLMO_MM_WORKERS" ]]; then
  data_args+=(olmo_mm_num_workers="$OLMO_MM_WORKERS")
fi

cat <<INFO
=== MaxText Olmo 3.5 multimodal SFT on $(basename "$CONFIG"), host $HOST ===
  run               : $OUTPUT_DIR/$RUN_NAME
  init weights      : $LOAD_PARAMETERS_PATH
  steps             : $STEPS (checkpoint every $CHECKPOINT_PERIOD)
  olmo_mm config    : ${OLMO_MM_CONFIG:-<from $CONFIG>}
  sharding          : ${SHARD_ARGS:-<from $CONFIG>}
  train args        : ${TRAIN_ARGS:-<none>}
  extra args        : ${EXTRA_ARGS:-<none>}
  LIBTPU_INIT_ARGS  : ${LIBTPU_INIT_ARGS:-<none>}
  TPU_MEGACORE      : ${TPU_MEGACORE:-<unset>}
  wandb             : $([[ -n "$WANDB_SECRET" ]] && echo "$WANDB_PROJECT/$RUN_NAME" || echo "<off>")
INFO
if [[ "${LIBTPU_INIT_ARGS:-}" == *xla_tpu_enable_async_ragged_all_to_all* ]]; then
  echo "ERROR: LIBTPU_INIT_ARGS must not set --xla_tpu_enable_async_ragged_all_to_all" >&2
  exit 1
fi

# Weights & Biases (key from Secret Manager through the node service account; the image has no wandb).
wandb_args=()
if [[ -n "$WANDB_SECRET" && -n "$DRY_RUN" ]]; then
  wandb_args=(enable_wandb=true wandb_project_name="$WANDB_PROJECT" wandb_run_name="$RUN_NAME")
elif [[ -n "$WANDB_SECRET" ]]; then
  pip list --format=freeze >/tmp/image-packages.txt
  pip install --quiet --root-user-action=ignore -c /tmp/image-packages.txt "wandb==$WANDB_VERSION"
  WANDB_API_KEY=$(python3 - "$WANDB_SECRET" <<'PY'
import base64, sys
import google.auth
from google.auth.transport.requests import AuthorizedSession
credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
response = AuthorizedSession(credentials).get(f"https://secretmanager.googleapis.com/v1/{sys.argv[1]}:access")
response.raise_for_status()
print(base64.b64decode(response.json()["payload"]["data"]).decode().strip())
PY
  )
  export WANDB_API_KEY
  export WANDB_RUN_ID="$RUN_NAME"
  wandb_args=(enable_wandb=true wandb_project_name="$WANDB_PROJECT" wandb_run_name="$RUN_NAME")
fi

# OLMO_ORBAX_BARRIER_TIMEOUT=N raises Orbax's multi-host barrier timeout from its fixed 300 s to N s: hosts that
# build the olmo_mm loader before the checkpoint load reach its barrier minutes after the others.
if [[ -z "$DRY_RUN" && -n "${OLMO_ORBAX_BARRIER_TIMEOUT:-}" ]]; then
  orbax_multihost=$(python3 -c 'import orbax.checkpoint.experimental.v1._src.synchronization.multihost as m; print(m.__file__)')
  sed -i "s/^_DEFAULT_BARRIER_TIMEOUT = .*/_DEFAULT_BARRIER_TIMEOUT = ${OLMO_ORBAX_BARRIER_TIMEOUT}/" "$orbax_multihost"
  grep -H '^_DEFAULT_BARRIER_TIMEOUT' "$orbax_multihost"
fi

# A host that aborts dumps core (about 75 GB) to the container's disk, and the kubelet then evicts the pod for
# ephemeral storage, which hides the real failure.
ulimit -c 0
export PYTHONUNBUFFERED=1
set -f # The overrides contain [..] lists; keep the shell from treating them as glob patterns.
# shellcheck disable=SC2086 # SHARD_ARGS, TRAIN_ARGS and EXTRA_ARGS are meant to split.
cmd=(python3 -m maxtext.trainers.pre_train.train "$CONFIG"
  run_name="$RUN_NAME"
  base_output_directory="$OUTPUT_DIR"
  load_parameters_path="$LOAD_PARAMETERS_PATH"
  steps="$STEPS"
  checkpoint_period="$CHECKPOINT_PERIOD"
  checkpoint_storage_use_ocdbt=false
  checkpoint_storage_use_zarr3=false
  ${data_args[@]+"${data_args[@]}"}
  ${wandb_args[@]+"${wandb_args[@]}"}
  $SHARD_ARGS
  $TRAIN_ARGS
  $EXTRA_ARGS)
if [[ -n "$DRY_RUN" ]]; then
  echo "DRY_RUN: OLMO_MM_LOADER_WORLD=${OLMO_MM_LOADER_WORLD:-<unset>} (cd /deps && ${cmd[*]})"
  exit 0
fi
cd /deps
set +e
"${cmd[@]}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
if [[ $rc -ne 0 ]]; then
  echo "train exited with $rc" >&2
  exit "$rc"
fi
# Guard against a clean exit with nothing trained (e.g. a data iterator that ends at once).
last=$(grep -oE 'completed step: [0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+$' || true)
if [[ -z "$last" ]]; then
  echo "STEP-GUARD: train exited 0 but no step completed" >&2
  exit 3
fi
echo "STEP-GUARD: last completed step $last (STEPS=$STEPS)"
