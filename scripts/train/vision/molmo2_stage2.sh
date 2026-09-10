#!/bin/bash
# Molmo2 stage-2 multimodal SFT — the merged single stage (docs/design/multimodal_sft.md).
#
# Full image-only-v9 mixture at Stage2-parity settings (Molmo2-4B, seq 16384,
# global 128 instances, LM 1e-5 / connector+vision 5e-6, compiled LM), with the
# nlp group produced by open-instruct's own tokenizer pipeline (Dolci-Instruct-SFT
# through the open_instruct_sft adapter) instead of the vision branch's weka dump.
#
# Usage (image name comes from build_image_and_launch.sh as $1):
#   ./scripts/train/build_image_and_launch.sh scripts/train/vision/molmo2_stage2.sh
#
# Init: HF allenai/Molmo2-4B by default. To start from a stage-1 OLMo-core
# checkpoint instead, append: --model_name_or_path /weka/path/to/stage1/run
#
# wandb: --with_tracking requires your per-user beaker secret in the workspace:
#   beaker secret write ${BEAKER_USER}_WANDB_API_KEY <key> --workspace ai2/open-instruct-dev
#
# Known upstream issues (see the design doc's findings):
# * compile_vision/compile_connector are OFF: their compiled backward hits an
#   inductor stride assertion when crop counts vary across batches.
# * The text source uses --text_chat_template_name olmo (system-turn-capable
#   ChatML): the Molmo2 tokenizer's built-in template rejects system messages.
set -euo pipefail

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "open-instruct-multimodal: Molmo2 stage-2 merged stage (image-only-v9 + Dolci via open_instruct_sft)." \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --gpus 8 \
    --no-host-networking \
    --no_auto_dataset_cache \
    --env OLMO2_FLEX_ATTN=1 \
    --env VIT_CROP_MICROBATCH=8 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -- torchrun --nproc_per_node=8 open_instruct/olmo_core_mixture_finetune.py \
    --exp_name molmo2_stage2 \
    --mixture image-only-v9 \
    --nlp_source open_instruct \
    --mixer_list allenai/Dolci-Instruct-SFT 1.0 \
    --text_chat_template_name olmo \
    --text_local_cache_dir /weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache \
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
    --output_dir "/weka/oe-adapt-default/allennlp/deletable_checkpoint/${BEAKER_USER}/molmo2_stage2"
