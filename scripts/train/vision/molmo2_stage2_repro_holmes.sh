#!/bin/bash
# Molmo2-4B stage-2 reproduction on ai2/holmes (B300) — capability validation.
#
# holmes needs CUDA 13 and we have no CUDA-13 image of the vision branch, so this
# uses the workspace's bootstrap pattern (cf. olmo-miles): launch inside a known
# CUDA-13 image, clone this branch at container start, and uv-sync the cuda13
# dependency group from the lockfile (cached on weka for fast restarts). mason joins
# the payload tokens into one `/bin/bash -c` string, so quoted '&&' tokens chain.
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
# A CUDA-13 environment image known to run on holmes in this workspace (olmo-miles);
# used only as the base OS/CUDA environment — code and python env are bootstrapped.
BOOTSTRAP_IMAGE="robertb/olmo-miles-v0-1-20260901"

echo "holmes validation: ${MAX_STEPS} steps, ref ${GIT_REF}, ckpt ${STAGE1_CKPT}"

uv run python mason.py \
    --cluster ai2/holmes \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BOOTSTRAP_IMAGE" \
    --description "open-instruct-multimodal: Molmo2-4B stage-2 repro on holmes (bootstrap, ${MAX_STEPS} steps)." \
    --pure_docker_mode \
    --num_nodes 1 \
    --gpus 8 \
    --no-host-networking \
    --no_auto_dataset_cache \
    --env OLMO2_FLEX_ATTN=1 \
    --env VIT_CROP_MICROBATCH=8 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -- \
    rm -rf /stage/oi '&&' \
    git clone --depth 1 -b "$GIT_REF" https://github.com/allenai/open-instruct.git /stage/oi '&&' \
    cd /stage/oi '&&' \
    bash scripts/train/vision/holmes_bootstrap.sh "$GIT_REF" \
    --nproc_per_node=8 open_instruct/olmo_core_mixture_finetune.py \
    --exp_name "molmo2_stage2_repro_4b_holmes_${MAX_STEPS}" \
    --mixture image-only-v9 \
    --model_name_or_path "$STAGE1_CKPT" \
    --compile_vision false \
    --compile_connector false \
    --max_train_steps "$MAX_STEPS" \
    --checkpointing_steps 2000 \
    --ephemeral_save_interval -1 \
    --keep_last_n_checkpoints -1 \
    --logging_steps 5 \
    --seed 6198 \
    --data_loader_seed 50189 \
    --with_tracking \
    --wandb_project molmo2-stage2 \
    --output_dir "/weka/oe-adapt-default/allennlp/deletable_checkpoint/${BEAKER_USER}/molmo2_stage2_repro_4b_holmes_${MAX_STEPS}"
