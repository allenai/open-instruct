#!/bin/bash
#
# Small-scale controlled-baseline tokenization. Runs the production olmo-hybrid
# mixer subsampled to $NUM_EXAMPLES rows. Tokenization code comes from whatever
# commit the Beaker image was built from.
#
# For a controlled A/B, launch twice from two separate image builds (origin/main
# worktree + HEAD) and compare the output dirs locally:
#   OUTPUT_SUFFIX=head \
#     ./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid/7b_think_sft_tokenization_small.sh
#   OUTPUT_SUFFIX=main \
#     ./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid/7b_think_sft_tokenization_small.sh
#   bash scripts/train/olmo-hybrid/_compare_tokenization.sh \
#     /weka/oe-adapt-default/$USER/dataset/olmo-hybrid-small-head \
#     /weka/oe-adapt-default/$USER/dataset/olmo-hybrid-small-main
#
set -euo pipefail

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-head}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1000000}"

TOKENIZER=allenai/dolma-2-tokenizer-olmo-3-instruct-final
tokenizer_path=/weka/oe-adapt-default/saumyam/open-instruct/dolma2-tokenizer-olmo-3-instruct-final
OUTPUT_DIR="/weka/oe-adapt-default/${BEAKER_USER}/dataset/olmo-hybrid-small-${OUTPUT_SUFFIX}"

uv run python mason.py \
  --cluster ai2/jupiter \
  --budget ai2/oe-adapt \
  --workspace ai2/olmo-instruct \
  --image "$BEAKER_IMAGE" \
  --pure_docker_mode \
  --no-host-networking \
  --gpus 0 \
  --priority urgent \
  --description "olmo-hybrid tokenize small (num_examples=$NUM_EXAMPLES, suffix=$OUTPUT_SUFFIX)" \
  --no_auto_dataset_cache \
  -- uv run python scripts/data/download_hf_repo.py --repo_id $TOKENIZER --local_dir $tokenizer_path \&\& uv run python scripts/data/convert_sft_data_for_olmocore.py \
      --dataset_mixer_list \
         allenai/Dolci-Think-SFT-32B 1.0 \
         allenai/olmo-toolu-sft-mix-T2-S2-f2-bfclv3-decontaminated-200K-thinking-id-fixed 3.0 \
         allenai/olmo-toolu-s2-sft-m3-thinking-id-fixed 3.0 \
         allenai/olmo-toolu-s2-sft-m4v2-thinking-id-fixed 3.0 \
         allenai/olmo-toolu-s2-sft-m5v2-thinking-id-fixed 3.0 \
         allenai/olmo-toolu_deepresearch_thinking_DRv4-modified-system-prompts 3.0 \
      --tokenizer_name_or_path $tokenizer_path \
      --output_dir "$OUTPUT_DIR" \
      --visualize True \
      --chat_template_name "olmo123" \
      --max_seq_length 32768 \
      --num_examples "$NUM_EXAMPLES"
