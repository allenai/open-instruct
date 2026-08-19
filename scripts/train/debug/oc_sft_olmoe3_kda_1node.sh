#!/bin/bash
# SFT spike for Jacob's OLMoE3 latent-KDA midtrain checkpoint (18.5B total /
# ~1.3B active; 20 layers, 16 KDA + 4 full-attention, 512 experts top-16,
# latent MoE dim 640, peri-LN, NoPE) over Dolci-Instruct-SFT, one 8-GPU node.
#
#   ./scripts/train/build_image_and_launch.sh --cuda-version 13 \
#       scripts/train/debug/oc_sft_olmoe3_kda_1node.sh train
#
# Unlike the s004 GDN spike this needs NO config migration -- the checkpoint was
# written by a compatible olmo-core, so its own config.json loads as-is (verified:
# zero unknown keys against OLMoDDPTransformerBlockConfig and OLMoDDPOptimizerConfig).
# We still pass --config_name explicitly because model_name_or_path is a DCP
# directory, and open-instruct requires a config for olmo-core checkpoints.
#
# Why the settings are what they are:
#
# * The tokenizer matches ours. Jacob's checkpoint and
#   allenai/olmo-3-tokenizer-instruct-dev are both GPT2Tokenizer at 100,278
#   tokens and agree on 100,268 of 100,274 shared tokens, producing identical ids
#   on ordinary text. The 100,352 in config.json is olmo-core's *padded* vocab,
#   not a different tokenizer. So the existing Dolci cache applies -- but the
#   cache key is not a function of the declared inputs alone (#1818), so confirm
#   the tokenize job is a cache hit rather than assuming it.
# * MAX_SEQ_LENGTH 8192 is the checkpoint's native context
#   (train_module.max_sequence_length), matching the s004 spike.
# * LR 2.5e-5, the Olmo Hybrid SFT rate. Midtraining ran at 1.6e-4 with the
#   branch's distributed skip-step AdamW; 2.5e-5 for SFT is the usual order-of-
#   magnitude step down but is not validated for this architecture.
# * These are nn.ddp MoE v2 models, so everything learned on the s004 spike
#   applies: OLMoDDPModel refuses FSDP2 and trains through the DDP train module,
#   async checkpointing is rejected, and the MoE permutation needs
#   transformer_engine 2.16.1 on the CUDA 13.3.1 image.
# * The checkpoint config carries ep degree 8; open-instruct builds no expert
#   parallel meshes, so the DDP train module config drops it. Full 18.5B
#   replication per rank -- fits on B300, not on H100.
# * Router load-balancing (lb_loss_weight 0.01) rides inside the model config,
#   so expert-collapse protection matches midtraining for free. Watch
#   "train/block NN/load imbalance" in W&B: on s004 it fell from ~1.87 to ~1.40
#   over 172 steps, and a sustained climb is the failure signal.

set -euo pipefail

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
MODE="${2:-train}"

# ---- cache-key arguments: MUST be byte-identical across both jobs ----
MODEL=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/midtraining/mt-1p2b-kda-ev2-neg-nope-gated-latentmoe-l2-paper-cx8-samebatch-lr1p6e-4-r1/step63802
# Not the checkpoint's own config: activation checkpointing is enabled and the
# expert-parallel block config dropped, because midtraining sharded the 512
# experts across 8 ranks and we replicate them. Regenerate with
# scripts/train/debug/make_kda_sft_config.py if the checkpoint moves.
CONFIG_NAME=scripts/train/debug/kda_mt_sft.json
TOKENIZER=allenai/olmo-3-tokenizer-instruct-dev
CHAT_TEMPLATE=olmo123
MAX_SEQ_LENGTH=8192
MIXER="allenai/Dolci-Instruct-SFT 1.0"
SEED=33333
LOCAL_CACHE_DIR=/weka/oe-adapt-default/allennlp/numpy_sft_cache
# ----------------------------------------------------------------------

# Global batch held at 1,048,576 tokens like every run in this series:
#   1 seq * 16 grad_accum * 8 ranks * 8192 tokens. 172 steps is ~0.1 epoch.
#   Smoke first: MAX_TRAIN_STEPS=30 CHECKPOINTING_STEPS=30.
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-172}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-86}"
DIST_TIMEOUT_HOURS="${DIST_TIMEOUT_HOURS:-4}"
# Mandatory: OLMoDDPTrainModule raises on async checkpointing.
SAVE_ASYNC_FLAG="--no_save_async"

echo "Using Beaker image: $BEAKER_IMAGE"
echo "Mode: $MODE"

if [[ "$MODE" == "tokenize" ]]; then
    uv run python mason.py \
        --cluster ai2/saturn ai2/neptune ai2/ceres \
        --workspace ai2/open-instruct-dev \
        --priority urgent \
        --image "$BEAKER_IMAGE" \
        --description "Tokenize Dolci-Instruct-SFT (seq 8192, olmo123) for the OLMoE3 KDA spike" \
        --pure_docker_mode \
        --preemptible \
        --num_nodes 1 \
        --gpus 0 \
        --non_resumable \
        --no_auto_dataset_cache \
        -- uv run python open_instruct/olmo_core_finetune.py \
        --model_name_or_path allenai/Olmo-3-1025-7B \
        --tokenizer_name_or_path $TOKENIZER \
        --chat_template_name $CHAT_TEMPLATE \
        --max_seq_length $MAX_SEQ_LENGTH \
        --mixer_list $MIXER \
        --local_cache_dir $LOCAL_CACHE_DIR \
        --seed $SEED \
        --cache_dataset_only
elif [[ "$MODE" == "train" ]]; then
    uv run python mason.py \
        --cluster ai2/holmes \
        --workspace ai2/open-instruct-dev \
        --priority urgent \
        --image "$BEAKER_IMAGE" \
        --description "OLMoE3 KDA midtrain SFT, Dolci-Instruct-SFT, $MAX_TRAIN_STEPS steps, 1x8 (seq 8192)" \
        --pure_docker_mode \
        --preemptible \
        --num_nodes 1 \
        --gpus 8 \
        --non_resumable \
        --no_auto_dataset_cache \
        --env OLMO_SHARED_FS=1 \
        -- torchrun \
        --nnodes=1 \
        --nproc_per_node=8 \
        open_instruct/olmo_core_finetune.py \
        --model_name_or_path $MODEL \
        --config_name $CONFIG_NAME \
        --tokenizer_name_or_path $TOKENIZER \
        --chat_template_name $CHAT_TEMPLATE \
        --max_seq_length $MAX_SEQ_LENGTH \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --learning_rate 2.5e-5 \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --max_grad_norm 1.0 \
        --num_epochs 1 \
        --max_train_steps $MAX_TRAIN_STEPS \
        --attn_implementation flash_2 \
        --checkpointing_steps $CHECKPOINTING_STEPS \
        --ephemeral_save_interval -1 \
        --keep_last_n_checkpoints -1 \
        --dist_timeout_hours $DIST_TIMEOUT_HOURS \
        $SAVE_ASYNC_FLAG \
        --with_tracking \
        --logging_steps 1 \
        --mixer_list $MIXER \
        --local_cache_dir $LOCAL_CACHE_DIR \
        --seed $SEED \
        --data_loader_seed 34521 \
        --output_dir \$CHECKPOINT_OUTPUT_DIR
else
    echo "Unknown mode: $MODE (expected 'tokenize' or 'train')" >&2
    exit 1
fi
