#!/bin/bash
#
# Debug script for SFT tokenization (~5 min test run)
# Usage: ./scripts/train/build_image_and_launch.sh scripts/train/debug/sft-tokenization.sh
#
set -euo pipefail

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
  --cluster ai2/jupiter \
  --cluster ai2/saturn \
  --cluster ai2/ceres \
  --budget ai2/oe-adapt \
  --workspace ai2/open-instruct-dev \
  --image "$BEAKER_IMAGE" \
  --pure_docker_mode \
  --no-host-networking \
  --gpus 1 \
  --priority urgent \
  --timeout 15m \
  --description "Debug SFT tokenization test" \
  --no_auto_dataset_cache \
  -- python scripts/data/convert_sft_data_for_olmocore.py \
      --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 0.01 \
      --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
      --output_dir /weka/oe-adapt-default/${BEAKER_USER}/dataset/sft-tokenization-debug \
      --visualize True \
      --chat_template_name olmo \
      --max_seq_length 8192 \
      --num_examples 1000 \
      --resume \
      --checkpoint_interval 500
