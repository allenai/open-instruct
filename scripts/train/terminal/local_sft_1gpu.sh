#!/bin/bash

# Local single-GPU debug run for terminal SFT.
# Uses Qwen3-0.6B + a single small dataset split instead of the full 32-GPU 9B job.
# No Beaker, no DeepSpeed, no mason.py -- just runs directly.

DATASET="hamishivi/tmax-sft-full-20260317"

uv run accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    open_instruct/finetune.py \
    --exp_name tmax_sft_local_1gpu \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --tokenizer_name Qwen/Qwen3-0.6B \
    --use_liger_kernel \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --dataset_mixer_list $DATASET 50 \
    --dataset_mixer_list_splits nvidia__Nemotron_Terminal_Corpus__dataset_adapters \
    --report_to none \
    --with_tracking \
    --logging_steps 1 \
    --seed 42 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
