#!/bin/bash
#
# Launch a single model merge job on Beaker
#
# Usage: ./scripts/merge/merge.sh <output_dir> <model1> <model2> [model3] ...
#
# Example:
#   ./scripts/merge/merge.sh /weka/oe-adapt-default/nathanl/merged/my-merge \
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

# Build the YAML config
CONFIG="models:"
for model in "${MODELS[@]}"; do
    CONFIG="$CONFIG
  - model: $model
    parameters:
      weight: 1.0"
done
CONFIG="$CONFIG
merge_method: linear
dtype: bfloat16"

# Base64 encode to avoid escaping issues
CONFIG_B64=$(echo "$CONFIG" | base64 | tr -d '\n')

# First model for tokenizer copy
FIRST_MODEL="${MODELS[0]}"

echo "=============================================="
echo "Launching ${NUM_MODELS}-model merge"
echo "Output: $OUTPUT_DIR"
echo "Models:"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo "=============================================="

# NOTE: MergeKit doesn't support hybrid models yet. Use linear_merge.sh for those.
#
# Custom installs needed for new architectures (uncomment and set PR number as needed)
# CUSTOM_INSTALL="uv pip install git+https://github.com/huggingface/transformers.git@refs/pull/XXXXX/head"
# INSTALL_CMD="${CUSTOM_INSTALL} && uv pip install mergekit"
INSTALL_CMD="uv pip install mergekit"

uv run python mason.py \
    --cluster ai2/jupiter \
    --budget ai2/oe-adapt \
    --workspace ai2/olmo-instruct \
    --image "$IMAGE" \
    --pure_docker_mode \
    --no-host-networking \
    --gpus 0 \
    --priority normal \
    --description "Model merge: ${NUM_MODELS}-model linear" \
    -- bash -c \"cd /stage \&\& $INSTALL_CMD \&\& echo $CONFIG_B64 \| base64 -d \> /tmp/c.yaml \&\& uv run mergekit-yaml /tmp/c.yaml $OUTPUT_DIR \&\& cp $FIRST_MODEL/tokenizer* $FIRST_MODEL/special_tokens_map.json $OUTPUT_DIR/ 2\>/dev/null\; test -f $FIRST_MODEL/chat_template.jinja \&\& cp $FIRST_MODEL/chat_template.jinja $OUTPUT_DIR/\; echo Done: $OUTPUT_DIR\"
