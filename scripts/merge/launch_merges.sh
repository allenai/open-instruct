#!/bin/bash
#
# Example: Launch multiple merge jobs
#
# Usage: ./scripts/merge/launch_merges.sh
#
set -euo pipefail

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')

MODEL_1="/weka/oe-adapt-default/nathanl/checkpoints/TEST_HYBRIC_SFT_LARGER_LR2.5e-5/step46412-hf"
MODEL_2="/weka/oe-adapt-default/nathanl/checkpoints/TEST_HYBRIC_SFT_LARGER_LR4.5e-5_seed42/step46412-hf"
MODEL_3="/weka/oe-adapt-default/nathanl/checkpoints/TEST_HYBRIC_SFT_LARGER_LR1e-5/step46412-hf"

# Use direct_merge.sh for new architectures (bypasses mergekit)
# Use mergekit_merge.sh for architectures supported by mergekit

# 2-model merge
./scripts/merge/direct_merge.sh \
    "/weka/oe-adapt-default/${BEAKER_USER}/merged/sft-2model-linear" \
    "$MODEL_1" "$MODEL_2"

# 3-model merge
./scripts/merge/direct_merge.sh \
    "/weka/oe-adapt-default/${BEAKER_USER}/merged/sft-3model-linear" \
    "$MODEL_1" "$MODEL_2" "$MODEL_3"

echo "All merge jobs launched!"
