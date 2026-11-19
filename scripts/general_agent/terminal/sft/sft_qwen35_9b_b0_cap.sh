#!/bin/bash
# B0-cap: capability-MATCHED control for the SafeComp smoke.
#
# Purpose: the attribution leg that holds the CAPABILITY half identical to the smoke and drops ONLY the
# 1k SafeComp safety rows. Comparing base -> B0-cap -> smoke isolates "does adding 1k safety help,
# holding capability fixed" (single variable). See agent-safety/evals/TRAINING_PLAN.md rung 3 (B0/B3).
#
# This is `sft_qwen35_9b_safecomp_smoke.sh` with exactly TWO changes:
#   1. SafeComp removed from --dataset_mixer_list (tmax-only; the capability half is unchanged)
#   2. --exp_name / --output_dir renamed for this control
# Everything else (base, LR, seq, SP, epochs=1, seed, image, no push/auto-eval) is kept identical so the
# capability half stays a faithful match to the smoke's.
#
# NB open-instruct amount convention: "0.2" (with a decimal) = 20% of the split (~3.3k). An integer like
# "1" would mean 1 SAMPLE (footgun). tmax 0.2 here == the smoke's capability half, exactly.
BEAKER_IMAGE="${1:-mingqianz/open-instruct-integration-test-agent-safety}"
echo "Using Beaker image: $BEAKER_IMAGE"

MODEL=hamishivi/Qwen3.5-9B
TOKENIZER=hamishivi/Qwen3.5-9B

# capability half (identical to the smoke)
DATASET=osieosie/tmax-sft-skill-tax-20260505-2.2k-combined-balanced-qwen3.6-27b-thinking
DATASET_CONFIG=skill_tax_20260505_2.2k_combined_balanced_thinking_all
CAP_FRAC=0.2

OUT=/weka/oe-adapt-default/mingqianz/models/qwen35-9b-b0-cap

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/oe-agents \
    --description "SFT Qwen3.5-9B: B0-cap capability-only control (tmax 0.2 ~3.3k, no safety)" \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --gpus 8 \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name sft_qwen35_9b_b0_cap \
    --model_name_or_path $MODEL \
    --tokenizer_name $TOKENIZER \
    --sequence_parallel_size 4 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --dataset_mixer_list \
        $DATASET $CAP_FRAC \
    --dataset_mixer_list_config_names \
        $DATASET_CONFIG \
    --dataset_mixer_list_splits \
        train \
    --output_dir $OUT \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --wandb_project_name oe-general-agents \
    --logging_steps 1 \
    --seed 42 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
