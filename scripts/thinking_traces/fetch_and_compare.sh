#!/usr/bin/env bash
#
# Download the result datasets of two thinking-trace Beaker experiments and run
# the cross-model comparison over them.
#
# Usage:
#   ./scripts/thinking_traces/fetch_and_compare.sh <exp-a> <exp-b> [outdir]
#
# Example:
#   ./scripts/thinking_traces/fetch_and_compare.sh 01ABC... 01DEF... results/think-len

set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,12p' "$0"
    exit 1
fi

EXP_A="$1"
EXP_B="$2"
OUTDIR="${3:-results/thinking_traces}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "$OUTDIR"
for exp in "$EXP_A" "$EXP_B"; do
    echo "=== fetching $exp ==="
    rm -rf "${OUTDIR:?}/$exp"
    beaker experiment results --output "$OUTDIR/$exp" "$exp"
done

# Each job writes exactly one traces_<served-name>.jsonl; the served name is what
# labels the model in the report, so derive the label from the filename rather
# than making the caller repeat it.
# No mapfile here: macOS still ships bash 3.2 and this half runs locally.
TRACE_FILES=()
while IFS= read -r f; do
    TRACE_FILES+=("$f")
done < <(find "$OUTDIR/$EXP_A" "$OUTDIR/$EXP_B" -name 'traces_*.jsonl' | sort)

if [ "${#TRACE_FILES[@]}" -lt 2 ]; then
    echo "error: expected 2 trace files, found ${#TRACE_FILES[@]}" >&2
    [ "${#TRACE_FILES[@]}" -gt 0 ] && printf '  %s\n' "${TRACE_FILES[@]}" >&2
    exit 1
fi

ARGS=()
for f in "${TRACE_FILES[@]}"; do
    label="$(basename "$f" .jsonl)"
    label="${label#traces_}"
    echo "  $label -> $f  ($(wc -l < "$f") traces)"
    ARGS+=(--traces "${label}=${f}")
done

cd "$REPO_ROOT"
PYTHONPATH=. uv run --no-project --python 3.11 --with numpy \
    python scripts/thinking_traces/analyze_traces.py \
    "${ARGS[@]}" --json-output "$OUTDIR/comparison.json" \
    | tee "$OUTDIR/comparison.txt"

echo
echo "wrote $OUTDIR/comparison.json and $OUTDIR/comparison.txt"
