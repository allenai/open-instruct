#!/bin/bash
#
# Usage: ./scripts/train/build_image_and_launch.sh scripts/train/olmo3/7b-hybrid-sft-tokenization.sh
#
set -euo pipefail
# Get the Beaker username to construct the image name
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

TOKENIZER=allenai/dolma-2-tokenizer-olmo-3-instruct-final
tokenizer_path=/weka/oe-adapt-default/saumyam/open-instruct/dolma2-tokenizer-olmo-3-instruct-final

uv run python mason.py \
  --cluster ai2/jupiter \
  --budget ai2/oe-adapt \
  --workspace ai2/olmo-instruct \
  --image "$BEAKER_IMAGE" \
  --pure_docker_mode \
  --no-host-networking \
  --gpus 8 \
  --priority urgent \
  --description "7B hybrid SFT tokenization" \
  --no_auto_dataset_cache \
  -- huggingface-cli download $TOKENIZER --local-dir $tokenizer_path \&\& python scripts/data/convert_sft_data_for_olmocore.py \
      --dataset_mixer_list \
         allenai/Dolci-Think-SFT-32B 1.0 \
         allenai/olmo-toolu-sft-mix-T2-S2-f2-bfclv3-decontaminated-200K-thinking-id-fixed 3.0 \
         allenai/olmo-toolu-s2-sft-m3-thinking-id-fixed 3.0 \
         allenai/olmo-toolu-s2-sft-m4v2-thinking-id-fixed 3.0 \
         allenai/olmo-toolu-s2-sft-m5v2-thinking-id-fixed 3.0 \
         allenai/olmo-toolu_deepresearch_thinking_DRv4-modified-system-prompts 3.0 \
      --tokenizer_name_or_path $tokenizer_path \
      --output_dir /weka/oe-adapt-default/nathanl/dataset/olmo-hybrid \
      --visualize True \
      --chat_template_name "olmo123" \
      --max_seq_length 32768
