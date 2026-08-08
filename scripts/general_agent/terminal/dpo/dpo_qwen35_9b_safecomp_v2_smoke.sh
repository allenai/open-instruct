#!/bin/bash
# SafeComp DPO smoke: Qwen3.5-9B SFT anchor + comp1k_v2 preference pairs, one 8-GPU node.
# Prereg: agent-safety/docs/experiments/2026-07-31-dpo-plan.md (frozen recipe: dpo_norm,
# beta 5.0, label_smoothing 0.0, no packing, seq 32768, seed 42).
#
# Rungs (prereg launch sequence):
#   CACHE_ONLY=dataset  bash <this>   # rung 3a — tokenize + cache the mixer, then exit (local OK)
#   CACHE_ONLY=logprobs bash <this>   # rung 3b — build the reference-logprob cache, then exit
#   bash <this>                       # rung 4  — full one-node optimizer/checkpoint/export smoke
#
# The tail-only transform (preference_safecomp_tail_tokenize_v1) labels exactly the divergent
# tail's assistant turns and DROPS pairs that exceed --max_seq_length untruncated (16/712 in
# comp1k_v2 -> 696 effective pairs). Global batch = 8 GPUs x 1 x accum 2 = 16 pairs/step
# -> ~43 optimizer steps for one epoch.
#
# NOTE the DPO flag names (differ from the SFT recipes): --mixer_list (not
# --dataset_mixer_list; no config-names equivalent, local jsonl only), --num_epochs,
# --tokenizer_name_or_path. Amount "1.0" = 100% of the file; "1" = ONE row.
set -euo pipefail

BEAKER_IMAGE="${1:-mingqianz/open-instruct-safecomp-dpo}"
MODEL=/weka/oe-adapt-default/mingqianz/models/qwen35-9b-safecomp-v2/sft_qwen35_9b_safecomp_v2
TOKENIZER=hamishivi/Qwen3.5-9B
DATA=/weka/oe-adapt-default/mingqianz/agent-safety/data/dpo/dpo_comp1k_v2.jsonl
OUT=/weka/oe-adapt-default/mingqianz/models/qwen35-9b-safecomp-v2-dpo-smoke
EXP_NAME=dpo_qwen35_9b_safecomp_v2_smoke
REF_CACHE=/weka/oe-adapt-default/mingqianz/models/.ref_logprobs_cache

CACHE_FLAG=""
case "${CACHE_ONLY:-}" in
  dataset)  CACHE_FLAG="--cache_dataset_only" ;;
  logprobs) CACHE_FLAG="--cache_logprobs_only" ;;
  "") ;;
  *) echo "CACHE_ONLY must be 'dataset', 'logprobs', or unset" >&2; exit 1 ;;
esac

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/oe-agents \
    --description "SafeComp DPO smoke: Qwen3.5-9B, 32k seq, 1 node ${CACHE_ONLY:+(cache: $CACHE_ONLY)}" \
    --priority urgent \
    --preemptible \
    --image "$BEAKER_IMAGE" --pure_docker_mode \
    --no_auto_dataset_cache \
    --env OLMO_SHARED_FS=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env REFERENCE_LOGPROBS_CACHE_PATH="$REF_CACHE" \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --gpus 8 -- torchrun --nproc_per_node=8 open_instruct/dpo.py \
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
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-7 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --seed 42 \
    --fsdp_shard_degree 8 \
    --fsdp_num_replicas 1 \
    --activation_checkpointing_mode selected_modules \
    --output_dir "$OUT" \
    --checkpointing_steps 20 \
    --logging_steps 1 \
    --with_tracking \
    --wandb_project oe-general-agents \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false \
    --try_auto_save_to_beaker false \
    $CACHE_FLAG
