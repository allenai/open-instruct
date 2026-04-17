#!/bin/bash
# Local test: finetune Qwen3.5-4B with packing to exercise the GatedDeltaNet packing fix.
# Runs two cases:
#   1. SP=1 (single GPU, non-SP packing — tests cu_seqlens fix)
#   2. SP=2 (two GPUs, Ulysses — tests CP context fix)
#
# Requires: 2x A100 GPUs, env -u LD_LIBRARY_PATH to avoid cuBLAS 12.9 conflict.
#
# Usage:
#   env -u LD_LIBRARY_PATH bash scripts/train/debug/qwen35_finetune_hybrid_test.sh

set -euo pipefail

export HF_HOME=/tmp/hf_home_qwen35_test
export HF_DATASETS_CACHE=/tmp/hf_home_qwen35_test/datasets
unset BEAKER_JOB_ID

MODEL="Qwen/Qwen3.5-4B"
DATASET="allenai/tulu-3-sft-personas-algebra"
OUTDIR="/tmp/qwen35_hybrid_test"

BASE_ARGS="
    open_instruct/finetune.py
    --model_name_or_path $MODEL
    --tokenizer_name $MODEL
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 1
    --learning_rate 1e-5
    --lr_scheduler_type constant
    --num_train_epochs 1
    --max_train_steps 3
    --logging_steps 1
    --dataset_mixer_list $DATASET 50
    --add_bos
    --seed 42
    --chat_template_name tulu
    --push_to_hub false
    --try_launch_beaker_eval_jobs false
    --dataset_local_cache_dir /tmp/qwen35_finetune_dataset_cache
"

echo "=========================================="
echo "Case 1: SP=1, packing (non-SP cu_seqlens fix)"
echo "=========================================="
env -u LD_LIBRARY_PATH /root/calzone/open-instruct/.venv/bin/accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    --dynamo_backend no \
    $BASE_ARGS \
    --max_seq_length 2048 \
    --packing \
    --output_dir "${OUTDIR}_sp1_pack"

echo ""
echo "=========================================="
echo "Case 2: SP=2, no packing (SP CP context fix)"
echo "=========================================="
env -u LD_LIBRARY_PATH /root/calzone/open-instruct/.venv/bin/accelerate launch \
    --mixed_precision bf16 \
    --num_processes 2 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --dynamo_backend no \
    $BASE_ARGS \
    --max_seq_length 512 \
    --sequence_parallel_size 2 \
    --output_dir "${OUTDIR}_sp2_nopack"

echo ""
echo "All cases passed."
