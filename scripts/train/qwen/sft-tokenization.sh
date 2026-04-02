#!/bin/bash
#
# Tokenize allenai/Dolci-Think-SFT-32B for Qwen3 SFT in OLMo-core.
#
# Produces 3 subsets: 10k, 100k, and full dataset.
# Uses the olmo_thinker chat template with a modified Qwen3 tokenizer
# (eos_token changed to <|endoftext|> for olmo-core bin-packing compatibility).
#
# Usage:
#   ./scripts/train/build_image_and_launch.sh scripts/train/qwen/sft-tokenization.sh
#
# Dataset Mixer Format:
#   --dataset_mixer_list takes pairs of [dataset_name, amount]:
#   - Float values (with decimal): proportion of dataset. "1.0" = 100%
#   - Integer values (no decimal): absolute sample count. "10000" = 10k samples
#
set -euo pipefail

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

echo "Using Beaker image: $BEAKER_IMAGE"

TOKENIZER_PATH=/weka/oe-adapt-default/jacobm/repos/cse-579/tokenizers/qwen3-olmo-thinker-eos
OUTPUT_BASE=/weka/oe-adapt-default/jacobm/repos/cse-579/datasets
DATASET=allenai/Dolci-Think-SFT-32B
TEMPLATE=olmo_thinker
MAX_SEQ=32768

# --- 10k subset ---
# echo "Launching 10k subset tokenization..."
# uv run python mason.py \
#   --cluster ai2/saturn \
#   --budget ai2/oe-adapt \
#   --workspace ai2/flex2 \
#   --image "$BEAKER_IMAGE" \
#   --no-host-networking \
#   --gpus 8 \
#   --priority urgent \
#   --description "Qwen3 SFT tokenization (10k)" \
#   --no_auto_dataset_cache \
#   --preemptible \
#   -- python scripts/data/prepare_modified_tokenizer.py --model Qwen/Qwen3-1.7B --save-dir ${TOKENIZER_PATH} --eos-token "'<|endoftext|>'" \&\& python scripts/data/convert_sft_data_for_olmocore.py \
#       --dataset_mixer_list ${DATASET} 10000 \
#       --tokenizer_name_or_path ${TOKENIZER_PATH} \
#       --output_dir ${OUTPUT_BASE}/Dolci-Think-SFT-32B-qwen3-olmo-thinker-10k \
#       --visualize True \
#       --chat_template_name "${TEMPLATE}" \
#       --max_seq_length ${MAX_SEQ}

# --- 100k subset ---
echo "Launching 100k subset tokenization..."
uv run python mason.py \
  --cluster ai2/saturn \
  --budget ai2/oe-adapt \
  --workspace ai2/flex2 \
  --image "$BEAKER_IMAGE" \
  --no-host-networking \
  --gpus 8 \
  --priority urgent \
  --description "Qwen3 SFT tokenization (100k)" \
  --no_auto_dataset_cache \
  --preemptible \
  -- python scripts/data/prepare_modified_tokenizer.py --model Qwen/Qwen3-1.7B --save-dir ${TOKENIZER_PATH} --eos-token "'<|endoftext|>'"" \&\& python scripts/data/convert_sft_data_for_olmocore.py \
      --dataset_mixer_list ${DATASET} 100000 \
      --tokenizer_name_or_path ${TOKENIZER_PATH} \
      --output_dir ${OUTPUT_BASE}/Dolci-Think-SFT-32B-qwen3-olmo-thinker-100k \
      --visualize True \
      --chat_template_name "${TEMPLATE}" \
      --max_seq_length ${MAX_SEQ}

# --- Full dataset ---
# echo "Launching full dataset tokenization..."
# uv run python mason.py \
#   --cluster ai2/saturn \
#   --budget ai2/oe-adapt \
#   --workspace ai2/flex2 \
#   --image "$BEAKER_IMAGE" \
#   --no-host-networking \
#   --gpus 8 \
#   --priority urgent \
#   --description "Qwen3 SFT tokenization (full)" \
#   --no_auto_dataset_cache \
#   --preemptible \
#   -- python scripts/data/prepare_modified_tokenizer.py --model Qwen/Qwen3-1.7B --save-dir ${TOKENIZER_PATH} --eos-token "'<|endoftext|>'"" \&\& python scripts/data/convert_sft_data_for_olmocore.py \
#       --dataset_mixer_list ${DATASET} 1.0 \
#       --tokenizer_name_or_path ${TOKENIZER_PATH} \
#       --output_dir ${OUTPUT_BASE}/Dolci-Think-SFT-32B-qwen3-olmo-thinker-full \
#       --visualize True \
#       --chat_template_name "${TEMPLATE}" \
#       --max_seq_length ${MAX_SEQ}

# echo "All 3 tokenization jobs launched."
