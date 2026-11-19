#!/bin/bash
# Benign-scope copy (2026-09-18) of the REGISTERED SafeComp DPO launcher — recipe byte-identical to
# dpo_safecomp_registered.sh; the only change is the MASON_PY submit-side interpreter switch below.
# Original prereg: agent-safety/docs/experiments/2026-07-31-dpo-plan.md.
# Parameterized ONLY over initial checkpoint, pair data, seed, and experiment tag (prereg
# "Required implementation"); the recipe below is frozen as validated by the 2026-08-08
# compatibility smoke (docs/experiments/2026-08-07-dpo-compat-smoke.md) and must not be edited
# per-run: dpo_norm, beta 5.0, label_smoothing 0.0 (default), no packing, seq 32768, lr 5e-7
# linear warmup 0.1, per-device 1 x accum 2 x 8 GPUs = 16 pairs/step, 1 epoch, fsdp shard 8,
# separate forwards + whole-block AC + compile off + chunked LM head (memory-validated).
#
# Usage (env-parameterized):
#   MODEL=<hf checkpoint> DATA=<pairs jsonl> SEED=<int> EXP_NAME=<tag> OUT=<dir> bash <this>
# Defaults = the registered seed-42 DPO treatment from the qwen35-9b-safecomp-v2 anchor.
set -euo pipefail

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"   # non-pure-docker: code+.venv from this weka repo
MODEL="${MODEL:-/weka/oe-adapt-default/mingqianz/models/qwen35-9b-safecomp-v2/sft_qwen35_9b_safecomp_v2}"
TOKENIZER="${TOKENIZER:-hamishivi/Qwen3.5-9B}"
DATA="${DATA:-/weka/oe-adapt-default/mingqianz/agent-safety/data/dpo/dpo_comp1k_v2.jsonl}"
SEED="${SEED:-42}"
EXP_NAME="${EXP_NAME:-dpo_qwen35_9b_safecomp_v2}"
OUT="${OUT:-/weka/oe-adapt-default/mingqianz/models/qwen35-9b-safecomp-v2-dpo}"
REF_CACHE="${REF_CACHE:-/weka/oe-adapt-default/mingqianz/models/.ref_logprobs_cache}"
EPOCHS="${EPOCHS:-1}"
# EXTRA_MASON_ARGS: extra mason flags, e.g. "--env HF_HOME=/tmp/hf_home --env HF_HUB_CACHE=/tmp/hf_home/hub" to load
# base weights from a node-local cache (2026-09-18: reading model.safetensors from the shared weka HF cache stalled for 35+ min).   # registered recipe = 1; the 2026-09-18 smoke-scale benign arms use 3 (documented deviation)

# MASON_PY: submit-side interpreter (default `uv run python`); uv stalls on this weka host (2026-09-18) ->
# MASON_PY=/tmp/claude-0/venv-sc/bin/python. Training itself runs .venv/bin/torchrun INSIDE the job (unchanged).
MASON_PY="${MASON_PY:-uv run python}"
PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}." $MASON_PY mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/oe-agents \
    --description "SafeComp DPO registered run: $EXP_NAME seed $SEED" \
    --priority urgent \
    --preemptible \
    --image "$BEAKER_IMAGE" \
    --no_auto_dataset_cache \
    --env OLMO_SHARED_FS=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env DPO_LOGPROB_CHUNK_SIZE=1024 \
    --env REFERENCE_LOGPROBS_CACHE_PATH="$REF_CACHE" \
    ${EXTRA_MASON_ARGS:-} \
    --num_nodes 1 \
    --gpus 8 -- .venv/bin/torchrun --nproc_per_node=8 open_instruct/dpo.py \
    --exp_name "$EXP_NAME" \
    --do_not_randomize_output_dir \
    --model_name_or_path "$MODEL" \
    --config_name qwen3_5_9B \
    --tokenizer_name_or_path "$TOKENIZER" \
    --mixer_list "$DATA" 1.0 \
    --transform_fn preference_safecomp_tail_tokenize_v1 preference_tulu_filter_v1 \
    --max_seq_length 32768 \
    --packing false \
    --loss_type dpo_norm \
    --beta 5.0 \
    --concatenated_forward false \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-7 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs "$EPOCHS" \
    --seed "$SEED" \
    --fsdp_shard_degree 8 \
    --fsdp_num_replicas 1 \
    --compile_model false \
    --activation_checkpointing_mode selected_modules \
    --activation_checkpointing_modules 'blocks.[0-9]' 'blocks.[0-9][0-9]' \
    --output_dir "$OUT" \
    --checkpointing_steps 20 \
    --logging_steps 1 \
    --with_tracking \
    --wandb_project oe-general-agents \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false \
    --try_auto_save_to_beaker false
