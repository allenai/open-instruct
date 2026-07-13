#!/bin/bash

# Local single-GPU debug run for terminal SFT.
# Uses Qwen3-0.6B + a single small dataset split instead of the full 32-GPU 9B job.
# No Beaker, no DeepSpeed, no mason.py -- just runs directly.

MODEL="Qwen/Qwen3-0.6B"
TOKENIZER="Qwen/Qwen3-0.6B"
DATASET="hamishivi/tmax-sft-full-20260317"
EXP_NAME="terminal_local_sft_tmax_1gpu"

uv run accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    open_instruct/finetune.py \
    --exp_name $EXP_NAME \
    --model_name_or_path $MODEL \
    --tokenizer_name $TOKENIZER \
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
    --checkpointing_steps epoch \
    --clean_checkpoints_at_end false \
    --dataset_mixer_list $DATASET 50 \
    --dataset_mixer_list_splits nvidia__Nemotron_Terminal_Corpus__dataset_adapters \
    --report_to none \
    --logging_steps 1 \
    --seed 42 \
    --report_to wandb \
    --with_tracking \
    --wandb_project_name oe-general-agents \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
