#!/usr/bin/env bash
#SBATCH --job-name=dolci-think-sft
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

# Dolci Think SFT training script.
# Trains OLMo-3 7B on the Dolci-Think-SFT dataset (~22B tokens).
#
# Prerequisites:
#   1. Clone OLMo-core: git clone https://github.com/allenai/OLMo-core.git
#   2. Prepare data: sbatch prepare_dolci_think_data.sh
#
# Usage:
#   mkdir -p logs
#   OLMOCORE_PATH=/path/to/OLMo-core \
#   DATASET_PATH=./data/dolci_think_sft_tokenized \
#   BASE_CKPT=/path/to/OLMo-3-7B \
#   sbatch train_dolci_think.sh

set -euo pipefail

# Validate required environment variables
if [[ -z "${OLMOCORE_PATH:-}" ]]; then
    echo "ERROR: OLMOCORE_PATH must be set to your OLMo-core clone"
    echo "  git clone https://github.com/allenai/OLMo-core.git"
    echo "  export OLMOCORE_PATH=/path/to/OLMo-core"
    exit 1
fi

if [[ -z "${DATASET_PATH:-}" ]]; then
    echo "ERROR: DATASET_PATH must be set to your tokenized dataset"
    echo "  First run: sbatch prepare_dolci_think_data.sh"
    exit 1
fi

if [[ -z "${BASE_CKPT:-}" ]]; then
    echo "ERROR: BASE_CKPT must be set to the base model checkpoint"
    exit 1
fi

# Add OLMo-core to Python path
export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"

# Think SFT defaults (from OLMo-3 paper Table 47)
RUN_NAME="${RUN_NAME:-dolci-think-sft}"
GPUS="${GPUS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"  # 5e-5 for Think
SEQ_LEN="${SEQ_LEN:-32768}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((SEQ_LEN * 32))}"  # ~1M tokens
SAVE_FOLDER="${SAVE_FOLDER:-./checkpoints/${RUN_NAME}}"

# GPU memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Dolci Think SFT Training ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "OLMo-core: $OLMOCORE_PATH"
echo "Dataset: $DATASET_PATH"
echo "Base checkpoint: $BASE_CKPT"
echo "Learning rate: $LEARNING_RATE"
echo "Sequence length: $SEQ_LEN"
echo "Global batch size: $GLOBAL_BATCH_SIZE tokens"
echo "Epochs: $NUM_EPOCHS"
echo "================================="

mkdir -p "$SAVE_FOLDER"

torchrun --nproc-per-node="$GPUS" \
  "${OLMOCORE_PATH}/src/scripts/train/sft/OLMo-sft.py" train \
  --run_name="$RUN_NAME" \
  --cluster=slurm \
  --model.model_path="$BASE_CKPT" \
  --dataset.name=dolci-think-sft \
  --dataset.paths="[${DATASET_PATH}/token_ids_part_*.npy]" \
  --dataset.label_mask_paths="[${DATASET_PATH}/labels_mask_part_*.npy]" \
  --dataset.sequence_length="$SEQ_LEN" \
  --save_folder="$SAVE_FOLDER" \
  --train_module.optim.lr="$LEARNING_RATE" \
  --train_module.scheduler.t_warmup=200 \
  --global_batch_size="$GLOBAL_BATCH_SIZE" \
  --batch_size_config.num_nodes=1 \
  --batch_size_config.gpus_per_node="$GPUS" \
  --trainer.max_epochs="$NUM_EPOCHS" \
  --save_interval_ephemeral=1000 \
  --canceled_check_interval=5

echo "=== Training complete ==="
echo "Checkpoints saved to: $SAVE_FOLDER"
