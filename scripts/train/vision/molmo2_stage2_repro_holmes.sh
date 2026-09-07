#!/bin/bash
# Molmo2-4B stage-2 reproduction on ai2/holmes (B300) — capability validation.
#
# holmes needs CUDA 13 and we have no CUDA-13 image of the vision branch, so this
# uses the workspace's bootstrap pattern (cf. olmo-miles): launch inside a known
# CUDA-13 image, clone this branch at container start, and uv-sync the cuda13
# dependency group from the lockfile (cached on weka for fast restarts). mason joins
# the payload tokens into one `/bin/bash -c` string, so quoted '&&' tokens chain.
#
# Microbatch stays at the default 2 (Stage2 parity): mb4 OOMs at seq 16384
# (261.7/267.7 GiB) and mb3 violates divisibility (per-rank batch of 8 instances
# splits only by 1/2/4/8). The remaining speed levers are more nodes, or the
# upstream compile_vision fix (OLMo-core#848).
#
# Defaults to a SHORT validation (100 steps) so it does not duplicate the full
# 20k-step run on the CUDA-12 clusters; pass a step count to override.
#
# Usage:
#   bash scripts/train/vision/molmo2_stage2_repro_holmes.sh [MAX_STEPS] [STAGE1_CKPT] [GIT_REF]
set -euo pipefail

MAX_STEPS="${1:-100}"
STAGE1_CKPT="${2:-/weka/oe-training-default/ai2-llm/checkpoints/jasonr/molmo2-stage1-4b-lossw-20260828}"
GIT_REF="${3:-vision-pr4}"
NUM_NODES="${4:-1}"
RESUME_DIR="${5:-}"  # optional: an output dir with checkpoints to resume from (latest step)
# A CUDA-13 environment image known to run on holmes in this workspace (olmo-miles);
# used only as the base OS/CUDA environment — code and python env are bootstrapped.
BOOTSTRAP_IMAGE="robertb/olmo-miles-v0-1-20260901"

# MM_COMPILE_VISION=1 enables torch.compile for the vision tower and connector
# (upstream's default, which we disable because of OLMo-core#848: inductor pads
# saved-activation strides to 128B, then the backward stride guard rejects the
# natural stride when crop counts change). TORCHINDUCTOR_COMPREHENSIVE_PADDING=0
# turns that padding off, which should make the compiled path usable.
# Beaker only sets the replica env vars for multi-replica jobs; at 1 node they
# expand to empty and torchrun rejects the rendezvous endpoint (cf.
# oc_sft_olmo3_7b_1node.sh). Single-node torchrun defaults to localhost.
if [[ "$NUM_NODES" -gt 1 ]]; then
    RDZV_ARGS=(--nnodes="$NUM_NODES" '--node_rank=$BEAKER_REPLICA_RANK' '--master_addr=$BEAKER_LEADER_REPLICA_HOSTNAME' --master_port=29400)
else
    RDZV_ARGS=(--nnodes=1)
fi

if [[ "${MM_COMPILE_VISION:-0}" == "1" ]]; then
    COMPILE_ARGS=(--compile_vision true --compile_connector true)
    COMPILE_ENV=(--env TORCHINDUCTOR_COMPREHENSIVE_PADDING=0)
else
    COMPILE_ARGS=(--compile_vision false --compile_connector false)
    COMPILE_ENV=()
fi

echo "holmes run: ${MAX_STEPS} steps, ${NUM_NODES} node(s), ref ${GIT_REF}, ckpt ${STAGE1_CKPT}"

uv run python mason.py \
    --cluster ai2/holmes \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --max_retries 3 \
    --image "$BOOTSTRAP_IMAGE" \
    --description "open-instruct-multimodal: Molmo2-4B stage-2 repro on holmes (bootstrap, ${MAX_STEPS} steps)." \
    --pure_docker_mode \
    --num_nodes "$NUM_NODES" \
    --gpus 8 \
    $([ "$NUM_NODES" -eq 1 ] && echo --no-host-networking) \
    --no_auto_dataset_cache \
    --env OLMO2_FLEX_ATTN=1 \
    --env VIT_CROP_MICROBATCH=16 \
    --env OLMO_SHARED_FS=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${COMPILE_ENV[@]}" \
    -- \
    rm -rf /stage/oi '&&' \
    git clone --depth 1 -b "$GIT_REF" https://github.com/allenai/open-instruct.git /stage/oi '&&' \
    cd /stage/oi '&&' \
    bash scripts/train/vision/holmes_bootstrap.sh "$GIT_REF" \
    "${RDZV_ARGS[@]}" \
    --nproc_per_node=8 open_instruct/olmo_core_mixture_finetune.py \
    --exp_name "molmo2_stage2_repro_4b_holmes_${MAX_STEPS}_n${NUM_NODES}" \
    --mixture image-only-v9 \
    --model_name_or_path "$STAGE1_CKPT" \
    "${COMPILE_ARGS[@]}" \
    --max_train_steps "$MAX_STEPS" \
    --checkpointing_steps 1000 \
    --ephemeral_save_interval -1 \
    --keep_last_n_checkpoints -1 \
    --logging_steps 5 \
    --prefetch_workers 8 \
    --seed 6198 \
    --data_loader_seed 50189 \
    ${RESUME_DIR:+--resume_from_checkpoint "$RESUME_DIR"} \
    --with_tracking \
    --wandb_project molmo2-stage2 \
    --output_dir "/weka/oe-adapt-default/allennlp/deletable_checkpoint/${BEAKER_USER}/molmo2_stage2_repro_4b_holmes_${MAX_STEPS}_n${NUM_NODES}"
