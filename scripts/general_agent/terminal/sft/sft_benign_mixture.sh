#!/bin/bash
# SFT launcher for the benign-condition SafeComp arms (2026-09-18), derived from sft_tmax9b_safety_mixture.sh
# with two changes: MODEL is a parameter (hub id — snapshot_download() rejects local paths), and the
# capability half is a LOCAL jsonl (successful rollouts of the TRAIN split only; the HF `_thinking_all`
# config includes held-out benchmark tasks and 51% failed episodes, so it is not used).
# Recipe frozen to the validated shape: LR 2e-5 linear, warmup 0.03, seq 32768, SP=4, per-device 1 x accum 8
# (effective batch 16), seed 42, 1 epoch, 1 node x 8 H100. Verify after launch: steps x 16 = rows trained.
#
# Amount convention (open-instruct footgun): "1.0" = 100% OF THE FILE, "1" = ONE sample; integer = exact rows.
#
# Usage:
#   MODEL=allenai/tmax-9b|hamishivi/Qwen3.5-9B SAFECOMP=<safety jsonl> [REPLAY=<replay jsonl> REPLAY_AMT=1.0|<int>]
#   [SAFE_AMT=1.0] EXP_NAME=<tag> OUT=<dir> bash <this> [image]
set -euo pipefail

BEAKER_IMAGE="${1:-mingqianz/open-instruct-integration-test-agent-safety}"
echo "Using Beaker image: $BEAKER_IMAGE"

MODEL="${MODEL:?set MODEL to a hub id (allenai/tmax-9b or hamishivi/Qwen3.5-9B)}"
TOKENIZER="${TOKENIZER:-hamishivi/Qwen3.5-9B}"
SAFECOMP="${SAFECOMP:?set SAFECOMP to the safety jsonl}"
SAFE_AMT="${SAFE_AMT:-1.0}"
REPLAY="${REPLAY:-}"
REPLAY_AMT="${REPLAY_AMT:-1.0}"
EXP_NAME="${EXP_NAME:?set EXP_NAME}"
OUT="${OUT:?set OUT}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
WORKSPACE="${WORKSPACE:-ai2/oe-agents}"
EPOCHS="${EPOCHS:-1}"
# EXTRA_MASON_ARGS: extra mason flags, e.g. "--env HF_HOME=/tmp/hf_home --env HF_HUB_CACHE=/tmp/hf_home/hub" to load
# base weights from a node-local cache (2026-09-18: reading model.safetensors from the shared weka HF cache stalled for 35+ min).   # frozen recipe = 1; the 2026-09-18 smoke-scale benign arms use 3 (documented deviation)

[ -s "$SAFECOMP" ] || { echo "missing safety jsonl: $SAFECOMP"; exit 1; }
if [[ "$SAFE_AMT" == "0" ]]; then
    # capability-only control: the same replay rows, no safety rows (attribution arm)
    [ -s "$REPLAY" ] || { echo "SAFE_AMT=0 requires REPLAY"; exit 1; }
    MIXER_LIST=("$REPLAY" "$REPLAY_AMT")
    MIXER_CONFIGS=("none")
    MIXER_SPLITS=("train")
    DESC="SFT $MODEL: capability-only replay $(basename "$REPLAY") x $REPLAY_AMT"
elif [[ -n "$REPLAY" ]]; then
    [ -s "$REPLAY" ] || { echo "missing replay jsonl: $REPLAY"; exit 1; }
    MIXER_LIST=("$REPLAY" "$REPLAY_AMT" "$SAFECOMP" "$SAFE_AMT")
    MIXER_CONFIGS=("none" "none")
    MIXER_SPLITS=("train" "train")
    DESC="SFT $MODEL: replay $(basename "$REPLAY") x $REPLAY_AMT + $(basename "$SAFECOMP") x $SAFE_AMT"
else
    MIXER_LIST=("$SAFECOMP" "$SAFE_AMT")
    MIXER_CONFIGS=("none")
    MIXER_SPLITS=("train")
    DESC="SFT $MODEL: safety-only $(basename "$SAFECOMP") x $SAFE_AMT"
fi
echo "$DESC"

# MASON_PY: interpreter for the submit-side mason.py (default `uv run python`). On this weka host uv can stall
# on import (2026-09-18); pass MASON_PY=/tmp/claude-0/venv-sc/bin/python (needs beaker-py, requests, rich,
# backoff) and run from the repo root so `open_instruct.launch_utils` resolves.
MASON_PY="${MASON_PY:-uv run python}"
PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}." $MASON_PY mason.py \
    --cluster "$CLUSTER" \
    --workspace "$WORKSPACE" \
    --description "$DESC" \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --no_auto_dataset_cache \
    ${EXTRA_MASON_ARGS:-} \
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
    --model_name_or_path "$MODEL" \
    --tokenizer_name "$TOKENIZER" \
    --sequence_parallel_size 4 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs "$EPOCHS" \
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
    # crashes at finetune.py:1115 before training. We serve from weka and evaluate ourselves.
