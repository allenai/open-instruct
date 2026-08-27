#!/bin/bash
# Multimodal SFT (Molmo stage 2) smoke test: 2 GPUs, debug mixture (tulu4 + text_vqa +
# chart_qa_weighted), 10 steps, compile off, HF init from allenai/Molmo2-4B.
#
# The multimodal datasets live on weka (MOLMO_DATA_DIR), so this must run on a weka
# cluster (mason auto-mounts both weka buckets there). This entry point has no
# pre-tokenization stage: the mixture tokenizes on the fly, so it is deliberately NOT
# in mason's OPEN_INSTRUCT_COMMANDS and --no_auto_dataset_cache is kept for intent.
#
# See docs/design/multimodal_sft.md.
set -euo pipefail

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Multimodal SFT (Molmo2 stage 2) smoke test, 2 GPUs." \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --gpus 2 \
    --non_resumable \
    --no-host-networking \
    --no_auto_dataset_cache \
    --env OLMO2_FLEX_ATTN=1 \
    --env VIT_CROP_MICROBATCH=16 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -- torchrun --nproc_per_node=2 open_instruct/olmo_core_mixture_finetune.py \
    --exp_name mm_sft_debug \
    --mixture debug \
    --max_train_steps 10 \
    --global_batch_instances 2 \
    --rank_microbatch_instances 1 \
    --checkpointing_steps 5 \
    --ephemeral_save_interval -1 \
    --keep_last_n_checkpoints 1 \
    --logging_steps 1 \
    --seed 123 \
    --output_dir "/weka/oe-adapt-default/allennlp/deletable_checkpoint/${BEAKER_USER}/mm_sft_debug"
# Compile stays ON (Stage2 production parity): FlexAttention without torch.compile
# runs in eager mode, which materializes enough intermediates at seq 16384 to OOM.
# 2 GPUs, not 1: Molmo2-4B's static training state (fp32 master params + fp32
# grads + bf16 compute copies) alone nearly fills one 80GB H100 — the trainer's
# dry-run batch OOMs regardless of sequence length. Two FSDP ranks shard it.
# No --with_tracking: wandb needs a per-user WANDB_API_KEY beaker secret
# (<user>_WANDB_API_KEY in the workspace), which a smoke test shouldn't require.
