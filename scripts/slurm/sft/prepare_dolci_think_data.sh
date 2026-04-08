#!/usr/bin/env bash
#SBATCH --job-name=prepare-dolci-think
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

# Converts allenai/Dolci-Think-SFT-7B dataset to OLMo-core tokenized format.
#
# Usage:
#   mkdir -p logs  # Create logs directory first
#   sbatch prepare_dolci_think_data.sh
#
# Resume after interruption:
#   Just resubmit the same script - it will automatically resume from checkpoint.
#
# Environment variables (optional):
#   OUTPUT_DIR: Where to save tokenized data (default: ./data/dolci_think_sft_tokenized)
#   TOKENIZER: HuggingFace tokenizer to use (default: allenai/Olmo-3-7B-Think-SFT)

set -euo pipefail

# Configuration - adjust paths as needed
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/dolci_think_sft_tokenized}"
TOKENIZER="${TOKENIZER:-allenai/Olmo-3-7B-Think-SFT}"

# HuggingFace cache directories (optional, but recommended to set)
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/models/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PROJECT_ROOT}/data/huggingface}"

echo "=== Dolci Think Data Preparation ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Tokenizer: $TOKENIZER"
echo "Output: $OUTPUT_DIR"
echo "====================================="

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT"

python scripts/data/convert_sft_data_for_olmocore.py \
  --tokenizer_name_or_path "$TOKENIZER" \
  --dataset_mixer_list allenai/Dolci-Think-SFT-7B 1.0 \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_length 32768 \
  --visualize True \
  --resume \
  --checkpoint_interval 50000

echo "=== Data preparation complete ==="
echo "Output saved to: $OUTPUT_DIR"
