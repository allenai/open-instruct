#!/bin/bash
# REGISTERED SafeComp DPO launcher — prereg: agent-safety/docs/experiments/2026-07-31-dpo-plan.md.
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
# EXTRA_MASON_ARGS (2026-10-06, f08): submit-side mason flags only, e.g. "--timeout 8h" (GPU teardown backstop).
# NUM_EPOCHS / CHECKPOINTING_STEPS (2026-10-07, f08 *-hi arms): unset = the registered 1 epoch / every 20 steps,
# so every registered arm resolves to the same command as before. Set only by f08_launch.sh *-hi (non-registered).
# MASON_PY / WORKSPACE (2026-10-11): see the mason call below. Backup: <this>.pre-alloc-20261011T212212Z.
# EPHEMERAL_SAVE_INTERVAL / KEEP_LAST_N_CHECKPOINTS (2026-10-25, Phase 0, f08 *-hi arms): unset = the flags are not
# passed at all (every earlier launch resolves byte-identically). Set by f08_launch.sh for a rolling olmo-core
# EPHEMERAL save (only the newest one is kept; permanent / epoch checkpoints are never touched by it) and keep-all
# permanent checkpoints (-1), so a preempted 2-epoch run resumes from <= 20 steps back and step<E> survives.
# Backup: <this>.bak-20261025T214316Z.
# Recipe-sweep flags (2026-10-26, post-ARR Phase 1, f08_launch.sh RS-* arms; NOT registered): CHOSEN_NLL_COEF (RPO NLL
# on chosen), HF_EXPORT_STEPS (space-separated steps -> <out>/hf_step<N>), HF_EXPORT_DTYPE, CHECKPOINT_SAVE_FOLDER (raw
# OLMo-core checkpoints elsewhere), PRE_TRAIN_CHECKPOINT (false = no step0). Unset = the flags are not passed at all,
# so every earlier launch resolves byte-identically. Backup: <this>.bak-20261026-recipesweep.
# Ablation flags (2026-11-03, f08_launch.sh v16 RS-{sftc,nll02,gtn,lo}-v21738 arms; NOT registered): DPO_LOSS_COEF (scale
# of the DPO term; 0 + CHOSEN_NLL_COEF 1 = SFT on chosen), TEMPLATE_NLL_COEF / TEMPLATE_NLL_TAU (confidence-gated NLL on
# chosen-side template tokens), LOG_TEMPLATE_PROBS (per-step template-token probabilities). Unset = not passed, so every
# earlier launch resolves byte-identically. Backup: <this>.bak-20261103T232400Z.
# Loss-weighting / clipping flags (2026-11-07, 2026-11-05 DPO code review findings 1 and 2; NOT registered):
# PAIR_AVG_LOSS (true = every pair weighs 1/global pairs instead of micro-batch token share), MAX_GRAD_NORM (OLMo-core's
# existing --max_grad_norm clipping; unset = no clipping, as before). Unset = not passed, so every earlier launch
# resolves byte-identically. PRIORITY (submit side only): Beaker job priority, unset = urgent as before.
# Backup: <this>.bak-20261107T214440Z.
set -euo pipefail

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"   # non-pure-docker: code+.venv from this weka repo
MODEL="${MODEL:-/weka/oe-adapt-default/mingqianz/models/qwen35-9b-safecomp-v2/sft_qwen35_9b_safecomp_v2}"
TOKENIZER="${TOKENIZER:-hamishivi/Qwen3.5-9B}"
DATA="${DATA:-/weka/oe-adapt-default/mingqianz/agent-safety/data/dpo/dpo_comp1k_v2.jsonl}"
SEED="${SEED:-42}"
EXP_NAME="${EXP_NAME:-dpo_qwen35_9b_safecomp_v2}"
OUT="${OUT:-/weka/oe-adapt-default/mingqianz/models/qwen35-9b-safecomp-v2-dpo}"
REF_CACHE="${REF_CACHE:-/weka/oe-adapt-default/mingqianz/models/.ref_logprobs_cache}"

# MASON_PY / WORKSPACE (2026-10-11, f08 alloc-* relaunch; submit side only, the recipe is unchanged): unset = the
# previous `uv run python` + ai2/oe-agents, so every existing launch resolves identically. A non-uv MASON_PY (a /tmp
# venv with beaker-py >= 2.8 for mason --min_runtime) gets this repo's .venv/bin prepended to PATH for the mason
# call only, so the job's PATH (mason forwards the submit-side PATH) matches a `uv run` launch.
MASON_PY="${MASON_PY:-uv run python}"
MASON_PATH="$PATH"; [ "$MASON_PY" = "uv run python" ] || MASON_PATH="$PWD/.venv/bin:$PATH"
PATH="$MASON_PATH" $MASON_PY mason.py \
    --cluster ${CLUSTER:-ai2/jupiter} \
    --workspace "${WORKSPACE:-ai2/oe-agents}" \
    --description "SafeComp DPO registered run: $EXP_NAME seed $SEED" \
    --priority "${PRIORITY:-urgent}" \
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
    --learning_rate "${LR:-5e-7}" \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs "${NUM_EPOCHS:-1}" \
    --seed "$SEED" \
    --fsdp_shard_degree 8 \
    --fsdp_num_replicas 1 \
    --compile_model false \
    --activation_checkpointing_mode selected_modules \
    --activation_checkpointing_modules 'blocks.[0-9]' 'blocks.[0-9][0-9]' \
    --output_dir "$OUT" \
    --checkpointing_steps "${CHECKPOINTING_STEPS:-20}" \
    ${EPHEMERAL_SAVE_INTERVAL:+--ephemeral_save_interval "$EPHEMERAL_SAVE_INTERVAL"} \
    ${KEEP_LAST_N_CHECKPOINTS:+--keep_last_n_checkpoints "$KEEP_LAST_N_CHECKPOINTS"} \
    ${CHOSEN_NLL_COEF:+--chosen_nll_coef "$CHOSEN_NLL_COEF"} \
    ${DPO_LOSS_COEF:+--dpo_loss_coef "$DPO_LOSS_COEF"} \
    ${TEMPLATE_NLL_COEF:+--template_nll_coef "$TEMPLATE_NLL_COEF"} \
    ${TEMPLATE_NLL_TAU:+--template_nll_tau "$TEMPLATE_NLL_TAU"} \
    ${LOG_TEMPLATE_PROBS:+--log_template_probs "$LOG_TEMPLATE_PROBS"} \
    ${PAIR_AVG_LOSS:+--pair_avg_loss "$PAIR_AVG_LOSS"} \
    ${MAX_GRAD_NORM:+--max_grad_norm "$MAX_GRAD_NORM"} \
    ${HF_EXPORT_STEPS:+--hf_export_steps $HF_EXPORT_STEPS} \
    ${HF_EXPORT_DTYPE:+--hf_export_dtype "$HF_EXPORT_DTYPE"} \
    ${CHECKPOINT_SAVE_FOLDER:+--checkpoint_save_folder "$CHECKPOINT_SAVE_FOLDER"} \
    ${PRE_TRAIN_CHECKPOINT:+--pre_train_checkpoint "$PRE_TRAIN_CHECKPOINT"} \
    --logging_steps 1 \
    --with_tracking \
    --wandb_project oe-general-agents \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false \
    --try_auto_save_to_beaker false
