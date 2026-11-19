#!/bin/bash
# Parameterized SFT launcher for the tmax-9b safety-mixture study
# (agent-safety/docs/experiments/2026-08-09-tmax9b-sft-dpo-safety-mixture-plan.md, small-set phase).
#
# Starting checkpoint is the FINAL tmax-9b (allenai/tmax-9b, local copy models/tmax-9b-servable —
# chat template + tokenizer verified byte-identical to hamishivi/Qwen3.5-9B, 2026-08-09).
# Recipe is frozen to the validated smoke/v2 shape: LR 2e-5 linear, warmup 0.03, seq 32768, SP=4,
# per-device 1 x accum 8 (effective batch 16), seed 42, 1 epoch. Change nothing per-run except
# SAFECOMP / CAP_FRAC / EXP_NAME / OUT.
#
# Amount convention (open-instruct footgun): "1.0" = 100% OF THE FILE, "1" = ONE sample. Do not
# drop the decimal point. Verify after launch by step arithmetic: steps x 16 = rows trained.
#
# Usage:
#   SAFECOMP=<safety jsonl> [CAP_FRAC=0.2] [SAFE_AMT=1.0] EXP_NAME=<tag> OUT=<dir> bash <this> [image]
#   CAP_FRAC unset or empty -> safety-only (no tmax capability half).
#   SAFE_AMT=0 -> capability-only (0% safety cell); CAP_FRAC required.
#   Amount convention applies to BOTH halves: decimal = fraction of split, integer = exact rows.
#   Grid cells (2026-08-09 larger-dataset extension) use integer exact counts for both halves.
set -euo pipefail

BEAKER_IMAGE="${1:-mingqianz/open-instruct-integration-test-agent-safety}"
echo "Using Beaker image: $BEAKER_IMAGE"

# Hub id, NOT the local servable path: this image's finetune.py pre-downloads via
# snapshot_download(), which rejects filesystem paths (HFValidationError, cost 4 failed
# launches 2026-08-09). Weights verified identical to models/tmax-9b-servable.
MODEL=allenai/tmax-9b
TOKENIZER=hamishivi/Qwen3.5-9B

# capability half — same source as the smoke/v2/B0 recipes
DATASET=osieosie/tmax-sft-skill-tax-20260505-2.2k-combined-balanced-qwen3.6-27b-thinking
DATASET_CONFIG=skill_tax_20260505_2.2k_combined_balanced_thinking_all

SAFECOMP="${SAFECOMP:?set SAFECOMP to the safety jsonl}"
CAP_FRAC="${CAP_FRAC:-}"
SAFE_AMT="${SAFE_AMT:-1.0}"
EXP_NAME="${EXP_NAME:?set EXP_NAME}"
OUT="${OUT:?set OUT}"

if [[ "$SAFE_AMT" == "0" ]]; then
    [[ -n "$CAP_FRAC" ]] || { echo "SAFE_AMT=0 requires CAP_FRAC"; exit 1; }
    MIXER_LIST=("$DATASET" "$CAP_FRAC")
    MIXER_CONFIGS=("$DATASET_CONFIG")
    MIXER_SPLITS=("train")
    DESC="SFT tmax-9b: capability-only (tmax $CAP_FRAC)"
elif [[ -n "$CAP_FRAC" ]]; then
    MIXER_LIST=("$DATASET" "$CAP_FRAC" "$SAFECOMP" "$SAFE_AMT")
    MIXER_CONFIGS=("$DATASET_CONFIG" "none")
    MIXER_SPLITS=("train" "train")
    DESC="SFT tmax-9b: capability(tmax $CAP_FRAC) + $(basename "$SAFECOMP") x $SAFE_AMT"
else
    MIXER_LIST=("$SAFECOMP" "$SAFE_AMT")
    MIXER_CONFIGS=("none")
    MIXER_SPLITS=("train")
    DESC="SFT tmax-9b: safety-only $(basename "$SAFECOMP") x $SAFE_AMT"
fi

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/oe-agents \
    --description "$DESC" \
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
    --exp_name "$EXP_NAME" \
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
    --dataset_mixer_list "${MIXER_LIST[@]}" \
    --dataset_mixer_list_config_names "${MIXER_CONFIGS[@]}" \
    --dataset_mixer_list_splits "${MIXER_SPLITS[@]}" \
    --output_dir "$OUT" \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --wandb_project_name oe-general-agents \
    --logging_steps 1 \
    --seed 42 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
    # try_launch_beaker_eval_jobs must stay false: this image lacks oe-eval-internal and the job
    # crashes at finetune.py:1115 before training. We run OAS ourselves and serve from weka.
