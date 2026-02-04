#!/bin/bash
#
# Launch a linear model merge job on Beaker (bypasses mergekit)
#
# This script directly averages safetensors weights, which works for
# new architectures that mergekit doesn't support yet (e.g., Olmo3_5Hybrid).
#
# Usage: ./scripts/merge/direct_merge.sh <output_dir> <model1> <model2> [model3] ...
#
# Example:
#   ./scripts/merge/direct_merge.sh /weka/oe-adapt-default/nathanl/merged/my-merge \
#       /weka/path/to/model1 \
#       /weka/path/to/model2
#
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <output_dir> <model1> <model2> [model3] ..."
    echo ""
    echo "Example:"
    echo "  $0 /weka/oe-adapt-default/user/merged/my-merge /weka/path/model1 /weka/path/model2"
    exit 1
fi

IMAGE="nathanl/open_instruct_auto"
OUTPUT_DIR="$1"
shift
MODELS=("$@")
NUM_MODELS=${#MODELS[@]}

# Build the models argument string
MODELS_ARG=""
for model in "${MODELS[@]}"; do
    MODELS_ARG="$MODELS_ARG $model"
done

echo "=============================================="
echo "Launching ${NUM_MODELS}-model linear merge"
echo "Output: $OUTPUT_DIR"
echo "Models:"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo "=============================================="

# Base64 encode the Python script to avoid escaping issues
SCRIPT_B64=$(base64 < scripts/merge/direct_merge.py | tr -d '\n')

uv run python mason.py \
    --cluster ai2/jupiter \
    --budget ai2/oe-adapt \
    --workspace ai2/olmo-instruct \
    --image "$IMAGE" \
    --pure_docker_mode \
    --no-host-networking \
    --gpus 0 \
    --priority normal \
    --description "Linear merge: ${NUM_MODELS}-model" \
    -- bash -c \"cd /stage \&\& echo $SCRIPT_B64 \| base64 -d \> /tmp/direct_merge.py \&\& uv run python /tmp/direct_merge.py --models $MODELS_ARG --output_dir $OUTPUT_DIR --dtype bfloat16\"
