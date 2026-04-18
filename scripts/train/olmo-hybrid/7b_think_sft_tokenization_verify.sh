#!/bin/bash
#
# Byte-for-byte verification harness for scripts/train/olmo-hybrid/7b_think_sft_tokenization.sh.
#
# Runs the hybrid SFT tokenization pipeline into a fresh output directory, then
# sha256-compares every tokenization artifact against a pre-existing reference
# directory produced by a prior (pre-refactor) run.
#
# Usage:
#   ./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid/7b_think_sft_tokenization_verify.sh
#
# Environment variables:
#   NEW_DIR       Where the new run writes (default: /weka/oe-adapt-default/$USER/dataset/olmo-hybrid-verify)
#   REFERENCE_DIR Reference artifacts to diff against (default: /weka/oe-adapt-default/$USER/dataset/olmo-hybrid)
#
set -euo pipefail

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

TOKENIZER=allenai/dolma-2-tokenizer-olmo-3-instruct-final
tokenizer_path=/weka/oe-adapt-default/saumyam/open-instruct/dolma2-tokenizer-olmo-3-instruct-final

NEW_DIR="${NEW_DIR:-/weka/oe-adapt-default/${BEAKER_USER}/dataset/olmo-hybrid-verify}"
REFERENCE_DIR="${REFERENCE_DIR:-/weka/oe-adapt-default/${BEAKER_USER}/dataset/olmo-hybrid}"

uv run python mason.py \
  --cluster ai2/jupiter \
  --budget ai2/oe-adapt \
  --workspace ai2/olmo-instruct \
  --image "$BEAKER_IMAGE" \
  --pure_docker_mode \
  --no-host-networking \
  --gpus 8 \
  --priority urgent \
  --description "7B hybrid SFT tokenization byte-for-byte verify" \
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
      --output_dir "$NEW_DIR" \
      --visualize True \
      --chat_template_name "olmo123" \
      --max_seq_length 32768 \&\& bash scripts/train/olmo-hybrid/_compare_tokenization.sh "$NEW_DIR" "$REFERENCE_DIR"
