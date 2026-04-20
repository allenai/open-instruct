#!/bin/bash
#
# Micro-benchmark of checkpoint-path wall-clock time: origin/main (JSON,
# full-state re-serialize per save) vs HEAD (incremental binary append).
#
# Usage: bash scripts/train/olmo-hybrid/benchmark_checkpoint.sh
#
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

NUM_EXAMPLES="${NUM_EXAMPLES:-5000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
WORKDIR="$(mktemp -d)"
MAIN_WORKTREE="$WORKDIR/main-worktree"
REF_DIR="$WORKDIR/ref"
NEW_DIR="$WORKDIR/new"

cleanup() {
  if [[ -d "$MAIN_WORKTREE" ]]; then
    git -C "$REPO_ROOT" worktree remove --force "$MAIN_WORKTREE" || true
  fi
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

TOKENIZER=allenai/OLMo-2-1124-7B

echo "=== workdir: $WORKDIR ==="
echo "=== num_examples=$NUM_EXAMPLES checkpoint_interval=$CHECKPOINT_INTERVAL ==="
git -C "$REPO_ROOT" worktree add --detach "$MAIN_WORKTREE" origin/main

COMMON_ARGS=(
  --dataset_mixer_list HuggingFaceH4/no_robots 1.0
  --dataset_mixer_list_splits train
  --tokenizer_name_or_path "$TOKENIZER"
  --chat_template_name tulu
  --max_seq_length 4096
  --shuffle_seed 42
  --num_examples "$NUM_EXAMPLES"
  --add_bos
  --checkpoint_interval "$CHECKPOINT_INTERVAL"
)

echo "=== [1/2] origin/main run ==="
start=$(date +%s)
(cd "$MAIN_WORKTREE" && uv run python scripts/data/convert_sft_data_for_olmocore.py \
  "${COMMON_ARGS[@]}" --output_dir "$REF_DIR") > "$WORKDIR/ref.log" 2>&1
ref_elapsed=$(( $(date +%s) - start ))
echo "origin/main wall time: ${ref_elapsed}s"

echo "=== [2/2] HEAD run ==="
start=$(date +%s)
(cd "$REPO_ROOT" && uv run python scripts/data/convert_sft_data_for_olmocore.py \
  "${COMMON_ARGS[@]}" --output_dir "$NEW_DIR") > "$WORKDIR/new.log" 2>&1
new_elapsed=$(( $(date +%s) - start ))
echo "HEAD wall time:        ${new_elapsed}s"

echo "=== summary ==="
echo "origin/main: ${ref_elapsed}s"
echo "HEAD:        ${new_elapsed}s"
if (( ref_elapsed > 0 )); then
  echo "speedup:     $(awk "BEGIN{printf \"%.2fx\", $ref_elapsed/$new_elapsed}")"
fi
