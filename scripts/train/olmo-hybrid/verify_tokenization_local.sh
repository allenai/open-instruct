#!/bin/bash
#
# Local byte-for-byte verification that the refactored
# open_instruct.numpy_dataset_conversion module (HEAD) produces identical
# tokenization output to scripts/data/convert_sft_data_for_olmocore.py on
# origin/main, for a small slice of the production mixer.
#
# Usage:
#   bash scripts/train/olmo-hybrid/verify_tokenization_local.sh
#
# Environment variables:
#   NUM_EXAMPLES   Number of rows to tokenize (default: 200)
#   WORKDIR        Override working directory (default: mktemp -d, auto-cleaned)
#
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

NUM_EXAMPLES="${NUM_EXAMPLES:-200}"
WORKDIR="${WORKDIR:-$(mktemp -d)}"
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
echo "=== creating throwaway worktree at origin/main ==="
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
)

echo "=== [1/3] reference run on origin/main ==="
(cd "$MAIN_WORKTREE" && uv run python scripts/data/convert_sft_data_for_olmocore.py \
  "${COMMON_ARGS[@]}" --output_dir "$REF_DIR")

echo "=== [2/3] new run on HEAD ==="
(cd "$REPO_ROOT" && uv run python scripts/data/convert_sft_data_for_olmocore.py \
  "${COMMON_ARGS[@]}" --output_dir "$NEW_DIR")

echo "=== [3/3] byte-for-byte compare ==="
bash "$REPO_ROOT/scripts/train/olmo-hybrid/_compare_tokenization.sh" "$NEW_DIR" "$REF_DIR"
