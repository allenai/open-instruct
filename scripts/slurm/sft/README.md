# SLURM Scripts for OLMo SFT

Example SLURM scripts for OLMo SFT (Supervised Fine-Tuning) on SLURM clusters.

## Overview

This directory contains scripts for:
1. **Data preparation** - Convert HuggingFace datasets to OLMo-core format
2. **Training** - Run SFT training using OLMo-core

## Prerequisites

### For Data Preparation
- open-instruct environment with dependencies installed

### For Training
Clone the OLMo-core repository:
```bash
git clone https://github.com/allenai/OLMo-core.git
export OLMOCORE_PATH=/path/to/OLMo-core
```

---

## Quick Start: Dolci Think SFT

```bash
mkdir -p logs

# 1. Prepare data (~12-24h)
sbatch prepare_dolci_think_data.sh

# 2. Train (~24h on 8x H100)
OLMOCORE_PATH=/path/to/OLMo-core \
DATASET_PATH=./data/dolci_think_sft_tokenized \
BASE_CKPT=/path/to/OLMo-3-7B \
sbatch train_dolci_think.sh
```

## Quick Start: Dolci Instruct SFT

```bash
mkdir -p logs

# 1. Prepare data (~6-12h)
sbatch prepare_dolci_instruct_data.sh

# 2. Train (~4h on 8x H100)
OLMOCORE_PATH=/path/to/OLMo-core \
DATASET_PATH=./data/dolci_instruct_sft_tokenized \
BASE_CKPT=/path/to/OLMo-3-7B \
sbatch train_dolci_instruct.sh
```

---

## Data Preparation Scripts

| Script | Dataset | Tokens | Time |
|--------|---------|--------|------|
| `prepare_dolci_think_data.sh` | `allenai/Dolci-Think-SFT-7B` | ~22B | 12-24h |
| `prepare_dolci_instruct_data.sh` | `allenai/Dolci-Instruct-SFT` | ~1.8B | 6-12h |

### Resume Support

Data preparation supports automatic resume after interruption:
- Checkpoints saved every 50,000 samples to `_checkpoint.json`
- Just resubmit the same script to resume
- Checkpoint is removed on successful completion

Useful for time-limited queues, preemptible instances, and fault tolerance.

### Configuration

```bash
OUTPUT_DIR=/path/to/output sbatch prepare_dolci_think_data.sh
HF_HOME=/scratch/hf_cache sbatch prepare_dolci_think_data.sh
```

---

## Training Scripts

| Script | Learning Rate | Time (8x H100) | Notes |
|--------|---------------|----------------|-------|
| `train_dolci_think.sh` | 5e-5 | ~24h | Larger dataset |
| `train_dolci_instruct.sh` | 8e-5 | ~4h | Smaller dataset |

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OLMOCORE_PATH` | Yes | - | Path to OLMo-core clone |
| `DATASET_PATH` | Yes | - | Path to tokenized dataset |
| `BASE_CKPT` | Yes | - | Base model checkpoint |
| `RUN_NAME` | No | `dolci-*-sft` | Experiment name |
| `LEARNING_RATE` | No | 5e-5 / 8e-5 | Learning rate |
| `SEQ_LEN` | No | `32768` | Sequence length |
| `NUM_EPOCHS` | No | `2` | Training epochs |
| `GPUS` | No | `8` | Number of GPUs |
| `SAVE_FOLDER` | No | `./checkpoints/$RUN_NAME` | Checkpoint directory |

---

## SLURM Customization

Edit the `#SBATCH` directives to match your cluster:

```bash
#SBATCH --partition=gpu          # Your GPU partition
#SBATCH --account=your-account   # Your allocation
#SBATCH --time=24:00:00          # Time limit
```

---

## Output Formats

### Tokenized Data
```
data/dolci_think_sft_tokenized/
├── tokenizer/                    # Saved tokenizer
├── token_ids_part_0000.npy       # Token IDs (~1GB chunks)
├── labels_mask_part_0000.npy     # Training labels
├── token_ids_part_0000.csv.gz    # Document boundaries
└── dataset_statistics.json       # Statistics
```

### Training Checkpoints
```
checkpoints/dolci-think-sft/
├── step-1000/
├── step-2000/
└── latest/
```

---

## Reference

- **OLMo-3 Paper**: [arxiv.org/abs/2512.13961](https://arxiv.org/abs/2512.13961) (Table 47 for hyperparameters)
- **OLMo-core**: [github.com/allenai/OLMo-core](https://github.com/allenai/OLMo-core)
- **Dolci Datasets**: [huggingface.co/allenai](https://huggingface.co/allenai)
