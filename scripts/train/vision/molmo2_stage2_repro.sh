#!/bin/bash
# Molmo2-4B stage-2 reproduction: upstream recipe, open-instruct entry point.
#
# Loads a stage-1 (caption-pretraining) OLMo-core checkpoint and trains stage 2 on
# the full image-only-v9 mixture with the UPSTREAM nlp source (tulu4 weka dump) —
# i.e. this reproduces mm_olmo / Molmo2-Stage2.py results through the new code, as
# opposed to molmo2_stage2.sh which swaps the nlp group for open-instruct data.
#
# Usage:
#   ./scripts/train/build_image_and_launch.sh scripts/train/vision/molmo2_stage2_repro.sh \
#       [STAGE1_CHECKPOINT_RUN_DIR]
# or with a prebuilt image:
#   bash scripts/train/vision/molmo2_stage2_repro.sh <beaker-image> [STAGE1_CHECKPOINT_RUN_DIR]
#
# The checkpoint arg accepts an OLMo-core run dir (latest step is used) or a step dir.
#
# Runs on ALLOCATED capacity (non-preemptible, normal priority) across the CUDA-12
# weka clusters. ai2/holmes (B300) needs a CUDA-13 image of this branch — build with
# ./scripts/train/build_image_and_launch.sh --cuda-version 13 before adding it here.
set -euo pipefail

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
STAGE1_CKPT="${2:-/weka/oe-training-default/ai2-llm/checkpoints/jasonr/molmo2-stage1-4b-lossw-20260828}"
echo "Using Beaker image: $BEAKER_IMAGE"
echo "Stage-1 checkpoint: $STAGE1_CKPT"

uv run python mason.py \
    --cluster ai2/jupiter ai2/saturn ai2/ceres \
    --workspace ai2/open-instruct-dev \
    --priority normal \
    --image "$BEAKER_IMAGE" \
    --description "open-instruct-multimodal: Molmo2-4B stage-2 reproduction (image-only-v9, stage-1 init, upstream recipe)." \
    --pure_docker_mode \
    --num_nodes 1 \
    --gpus 8 \
    --no-host-networking \
    --no_auto_dataset_cache \
    --env OLMO2_FLEX_ATTN=1 \
    --env VIT_CROP_MICROBATCH=8 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -- torchrun --nproc_per_node=8 open_instruct/olmo_core_mixture_finetune.py \
    --exp_name molmo2_stage2_repro_4b \
    --mixture image-only-v9 \
    --model_name_or_path "$STAGE1_CKPT" \
    --compile_vision false \
    --compile_connector false \
    --max_train_steps 20000 \
    --checkpointing_steps 2000 \
    --ephemeral_save_interval -1 \
    --keep_last_n_checkpoints -1 \
    --logging_steps 5 \
    --seed 6198 \
    --data_loader_seed 50189 \
    --with_tracking \
    --wandb_project molmo2-stage2 \
    --output_dir "/weka/oe-adapt-default/allennlp/deletable_checkpoint/${BEAKER_USER}/molmo2_stage2_repro_4b"
