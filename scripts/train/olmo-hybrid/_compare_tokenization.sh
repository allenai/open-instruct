#!/bin/bash
#
# Byte-for-byte comparison of two tokenization output directories produced by
# scripts/data/convert_sft_data_for_olmocore.py. Exits non-zero on any mismatch.
#
# Usage: _compare_tokenization.sh NEW_DIR REFERENCE_DIR

set -euo pipefail

NEW_DIR="$1"
REFERENCE_DIR="$2"

if [[ ! -d "$NEW_DIR" ]]; then
  echo "NEW_DIR does not exist: $NEW_DIR" >&2
  exit 1
fi
if [[ ! -d "$REFERENCE_DIR" ]]; then
  echo "REFERENCE_DIR does not exist: $REFERENCE_DIR" >&2
  exit 1
fi

echo "=== Comparing $NEW_DIR vs $REFERENCE_DIR ==="

mismatches=0

hash_file() {
  sha256sum "$1" | awk '{print $1}'
}

hash_stats_json() {
  jq -S 'del(.timestamp, .output_directory)' "$1" | sha256sum | awk '{print $1}'
}

compare_file() {
  local rel="$1"
  local new_path="$NEW_DIR/$rel"
  local ref_path="$REFERENCE_DIR/$rel"

  if [[ ! -f "$new_path" ]]; then
    echo "MISSING in new: $rel"
    mismatches=$((mismatches + 1))
    return
  fi
  if [[ ! -f "$ref_path" ]]; then
    echo "MISSING in reference: $rel"
    mismatches=$((mismatches + 1))
    return
  fi

  local new_hash ref_hash
  if [[ "$rel" == "dataset_statistics.json" ]]; then
    new_hash=$(hash_stats_json "$new_path")
    ref_hash=$(hash_stats_json "$ref_path")
  else
    new_hash=$(hash_file "$new_path")
    ref_hash=$(hash_file "$ref_path")
  fi

  if [[ "$new_hash" == "$ref_hash" ]]; then
    echo "OK   $rel ($new_hash)"
  else
    echo "DIFF $rel"
    echo "  new:  $new_hash  ($(stat -c %s "$new_path" 2>/dev/null || wc -c < "$new_path") bytes)"
    echo "  ref:  $ref_hash  ($(stat -c %s "$ref_path" 2>/dev/null || wc -c < "$ref_path") bytes)"
    if [[ "$rel" != "dataset_statistics.json" ]]; then
      echo "  first diverging bytes:"
      cmp -l "$new_path" "$ref_path" | head -20 || true
    fi
    mismatches=$((mismatches + 1))
  fi
}

mapfile -t new_files < <(cd "$NEW_DIR" && ls | sort)
mapfile -t ref_files < <(cd "$REFERENCE_DIR" && ls | sort)

echo "--- File listings ---"
echo "new:       ${new_files[*]}"
echo "reference: ${ref_files[*]}"

for f in "${new_files[@]}"; do
  case "$f" in
    token_ids_part_*.npy|labels_mask_part_*.npy|token_ids_part_*.csv.gz|dataset_statistics.json)
      compare_file "$f"
      ;;
    dataset_statistics.txt|tokenizer|_checkpoint.json|_checkpoint.json.tmp)
      echo "SKIP $f (not compared)"
      ;;
    *)
      echo "SKIP $f (unknown artifact, not compared)"
      ;;
  esac
done

if (( mismatches > 0 )); then
  echo "=== FAILED: $mismatches mismatch(es) ==="
  exit 1
fi

echo "=== PASSED: byte-for-byte match ==="
