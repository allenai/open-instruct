#!/bin/bash
# OLMo3-7B SFT Training on Tillicum (2-node × 8 GPUs = 16 GPUs)
#
# This is a Tillicum/Slurm conversion for OLMo3 7B SFT training.
# Uses standard open_instruct/finetune.py instead of custom training scripts.
# Scaled to 2 nodes (16 GPUs) to fit normal QOS limits.
#
# Requirements:
# - QOS 'normal' (no special approval needed)
# - Model: allenai/OLMo-2-1124-7B (base model for SFT)
# - HuggingFace token: Set HF_TOKEN environment variable
# - WandB token: Set WANDB_API_KEY environment variable
#
# Usage:
#   # First ensure tokens are set (one-time setup):
#   export HF_TOKEN="hf_your_token_here"
#   export WANDB_API_KEY="your_wandb_key_here"
#
#   # Then run:
#   bash scripts/train/tillicum/olmo3_7b_instruct_sft_2node.sh
#
# Estimated cost: 16 GPUs × 8 hours × $0.90/GPU-hour = $115.20
# Expected runtime: ~8 hours (depends on dataset size and epochs)
#
# Outputs:
# - Logs: /gpfs/scrubbed/$USER/experiments/olmo3_7b_sft_2node_*/logs/
# - Model: /gpfs/scrubbed/$USER/experiments/olmo3_7b_sft_2node_*/output/
# - Checkpoints: /gpfs/scrubbed/$USER/experiments/olmo3_7b_sft_2node_*/checkpoints/

# Check for required tokens
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set!"
    echo ""
    echo "Please set your HuggingFace token before running this script:"
    echo "  export HF_TOKEN='hf_your_token_here'"
    echo ""
    echo "To make this permanent, add to your ~/.bashrc or ~/.zshrc:"
    echo "  echo 'export HF_TOKEN=\"hf_your_token_here\"' >> ~/.bashrc"
    echo "  source ~/.bashrc"
    echo ""
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is not set!"
    echo ""
    echo "Please set your Weights & Biases API key before running this script:"
    echo "  export WANDB_API_KEY='your_wandb_key_here'"
    echo ""
    echo "To make this permanent, add to your ~/.bashrc or ~/.zshrc:"
    echo "  echo 'export WANDB_API_KEY=\"your_wandb_key_here\"' >> ~/.bashrc"
    echo "  source ~/.bashrc"
    echo ""
    exit 1
fi

echo "✓ Tokens verified: HF_TOKEN and WANDB_API_KEY are set"
echo ""

# Model configuration
# Note: Use OLMo-2-1124-7B as base model, or specify a different checkpoint
MODEL_NAME=${MODEL_PATH:-allenai/OLMo-2-1124-7B}

# Learning rate (original olmo-core: 8e-5, halved for open-instruct: 4e-5)
LR=4e-5

# Experiment name
EXP_NAME=olmo3-7b-instruct-SFT-tillicum-2node-${LR}

uv run python tillicum.py \
    --gpus 8 \
    --nodes 2 \
    --time 08:00:00 \
    --qos normal \
    --job_name olmo3_7b_sft_2node \
    --module gcc/13.4.0 \
    --module cuda/13.0.0 \
    "$@" \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name $EXP_NAME \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --use_slow_tokenizer False \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate $LR \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --logging_steps 1 \
    --dataset_mixer_list allenai/tulu-3-sft-mixture 1.0 \
    --add_bos \
    --seed 543210 \
    --chat_template_name olmo123 \
    --use_flash_attn \
    --gradient_checkpointing \
    --with_tracking \
    --output_dir '$EXPERIMENT_DIR/output' \
    --dataset_local_cache_dir '$DATASET_LOCAL_CACHE_DIR'
    # NOTE: Original script used a custom dataset and training script that are not
    # available in the current codebase. This script uses the standard finetune.py
    # with tulu-3-sft-mixture dataset. Adjust dataset_mixer_list as needed.
    #
    # Original configuration for reference (olmo-core):
    # - Dataset: gs://ai2-llm/jacobm/data/sft/rl-sft-32k/olmo3-32b-instruct-sft-1114
    # - Initial checkpoint: gs://ai2-llm/jacobm/checkpoints/olmo3-7b-reasoning-sft-final
    # - Nodes: 4 (32 GPUs)
    # - Global batch size: 1048576 tokens
    # - Seq len: 32768
    # - Epochs: 2
    # - Learning rate: 8e-5 (halved to 4e-5 for open-instruct)
