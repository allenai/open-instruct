#!/bin/bash
# Launch all backend A/B benchmark runs. Usage:
#   ./scripts/train/build_image_and_launch.sh scripts/train/debug/backend_ab/launch_all.sh
set -eo pipefail
BEAKER_IMAGE="${1:?usage: launch_all.sh BEAKER_IMAGE}"
DIR="$(dirname "$0")"

echo "=== Launching SFT numpy cache job (blocks ab_sft_olmocore) ==="
CACHE_LOG=$(bash "$DIR/ab_sft_olmocore_cache.sh" "$BEAKER_IMAGE" 2>&1 | tee /dev/stderr)
CACHE_ID=$(echo "$CACHE_LOG" | grep -oE 'https://beaker.org/ex/[a-zA-Z0-9]+' | tail -1 | sed 's|.*/||')
[ -n "$CACHE_ID" ] || { echo "ERROR: no cache experiment id"; exit 1; }

echo "=== Launching the 5 non-blocked runs ==="
bash "$DIR/ab_sft_deepspeed.sh" "$BEAKER_IMAGE"
bash "$DIR/ab_dpo_deepspeed.sh" "$BEAKER_IMAGE"
bash "$DIR/ab_dpo_olmocore.sh" "$BEAKER_IMAGE"
bash "$DIR/ab_grpo_deepspeed.sh" "$BEAKER_IMAGE"
bash "$DIR/ab_grpo_olmocore.sh" "$BEAKER_IMAGE"

echo "=== Waiting for cache job $CACHE_ID ==="
beaker experiment await-all "$CACHE_ID"
echo "=== Launching ab_sft_olmocore ==="
bash "$DIR/ab_sft_olmocore.sh" "$BEAKER_IMAGE"
echo "=== All launched ==="
