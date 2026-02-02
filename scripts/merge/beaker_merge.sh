#!/bin/bash
#
# Run model merging on Beaker (for weka-accessible models)
#
# Usage: ./scripts/train/build_image_and_launch.sh scripts/merge/beaker_merge.sh
#
set -euo pipefail

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

# ============================================================================
# Configuration - EDIT THESE
# ============================================================================

# Models to merge (edit paths as needed)
MODEL_1="/weka/oe-adapt-default/nathanl/checkpoints/TEST_HYBRIC_SFT_LARGER_LR2.5e-5/step46412-hf"
MODEL_2="/weka/oe-adapt-default/nathanl/checkpoints/TEST_HYBRIC_SFT_LARGER_LR4.5e-5_seed42/step46412-hf"
# MODEL_3="/weka/oe-adapt-default/nathanl/checkpoints/THIRD_MODEL/step46412-hf"

# Output directory
OUTPUT_DIR="/weka/oe-adapt-default/${BEAKER_USER}/merged/sft-2model-merge-$(date +%Y%m%d)"

# ============================================================================
# Launch merge job
# ============================================================================

uv run python mason.py \
    --cluster ai2/jupiter \
    --budget ai2/oe-adapt \
    --workspace ai2/olmo-instruct \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --no-host-networking \
    --gpus 0 \
    --priority normal \
    --description "Model merge: 2-model linear merge" \
    -- pip install mergekit \&\& python -c "
import yaml
import subprocess
import shutil
import os

# Create merge config
config = {
    'models': [
        {'model': '${MODEL_1}', 'parameters': {'weight': 1.0}},
        {'model': '${MODEL_2}', 'parameters': {'weight': 1.0}},
    ],
    'merge_method': 'linear',
    'dtype': 'bfloat16'
}

config_path = '/tmp/merge_config.yaml'
with open(config_path, 'w') as f:
    yaml.dump(config, f)

print('Merge config:')
print(yaml.dump(config))

# Run merge
output_dir = '${OUTPUT_DIR}'
os.makedirs(output_dir, exist_ok=True)

print(f'Running merge to {output_dir}...')
subprocess.run(['mergekit-yaml', config_path, output_dir], check=True)

# Copy tokenizer from first model if not present
tokenizer_files = ['tokenizer.json', 'tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json']
source_model = '${MODEL_1}'
for f in tokenizer_files:
    src = os.path.join(source_model, f)
    dst = os.path.join(output_dir, f)
    if os.path.exists(src) and not os.path.exists(dst):
        print(f'Copying {f}...')
        shutil.copy(src, dst)

print(f'Merge complete! Output at: {output_dir}')
"
