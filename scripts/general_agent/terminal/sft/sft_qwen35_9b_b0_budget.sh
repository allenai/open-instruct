#!/bin/bash
# B0-budget: token-budget-MATCHED control for the SafeComp smoke.
#
# Purpose: the attribution leg that spends the SAME total SFT budget as the smoke's mixture
# (3.3k tmax + 1k safety ~= 4.3k rows) entirely on capability data. Comparing B0-budget vs smoke
# isolates "does the safety data beat an EQUAL budget of capability tokens" — controls for the
# "any ~4k of SFT would help" confound. See agent-safety/evals/TRAINING_PLAN.md rung 3 (B0/B3, "matched
# token budget").
#
# This is `sft_qwen35_9b_safecomp_smoke.sh` with exactly TWO changes:
#   1. SafeComp removed and the tmax amount set to an EXACT COUNT of 4000 rows (tmax-only)
#   2. --exp_name / --output_dir renamed for this control
#
# CRITICAL open-instruct amount footgun (dataset_transformation.py L2299-2331): an integer WITHOUT a
# decimal point is a raw SAMPLE COUNT, a value WITH a decimal is a PROPORTION. So `4000` == exactly 4000
# rows (what we want); `4000.0` or `0.24` would be fractions. Do NOT add a decimal here. 4000 < split
# size, so the loop takes exactly int(4000) rows (reproducible via dataset_config_seed = --seed 42).
BEAKER_IMAGE="${1:-mingqianz/open-instruct-integration-test-agent-safety}"
echo "Using Beaker image: $BEAKER_IMAGE"

MODEL=hamishivi/Qwen3.5-9B
TOKENIZER=hamishivi/Qwen3.5-9B

# capability data, taken as an EXACT COUNT (no decimal) to hit the mixture's total budget
DATASET=osieosie/tmax-sft-skill-tax-20260505-2.2k-combined-balanced-qwen3.6-27b-thinking
DATASET_CONFIG=skill_tax_20260505_2.2k_combined_balanced_thinking_all
CAP_COUNT=4000

OUT=/weka/oe-adapt-default/mingqianz/models/qwen35-9b-b0-budget

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/oe-agents \
    --description "SFT Qwen3.5-9B: B0-budget capability-only control (tmax exact 4000, no safety)" \
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
    --exp_name sft_qwen35_9b_b0_budget \
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
        $DATASET $CAP_COUNT \
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
