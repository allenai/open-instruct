#!/bin/bash
# SFT spike for OLMoE3-dev-260614-s004 (26.7B total / 2.9B active MoE + GDN hybrid,
# 128 experts top-8 + 1 shared, peri-LN) over Dolci-Instruct-SFT, one 8-GPU node.
#
# PURPOSE: infra check ("does the open-instruct olmo-core path train this at all")
# and an expert-collapse check (watch the router lb loss in W&B). NOT an eval run:
# no HF export exists for this architecture and vLLM has no Olmo3MoeForCausalLM,
# so checkpoints from this run cannot be evaluated yet.
#
# RUN AS TWO JOBS (tokenize is a cache hit if 01M034QBWSKPK9W1JEXRHRER27 ran):
#
#   ./scripts/train/build_image_and_launch.sh --cuda-version 13 \
#       scripts/train/debug/oc_sft_s004_moe_1node.sh tokenize
#   ./scripts/train/build_image_and_launch.sh --cuda-version 13 \
#       scripts/train/debug/oc_sft_s004_moe_1node.sh train
#
# How this run is wired, none of it obvious:
#
# * MODEL is an olmo-core DCP checkpoint (pretrain step69000), not HF. It was
#   copied from gs://ai2-llm/checkpoints/olmo3moe/OLMoE3-dev-260614-s004_*/step69000
#   to WEKA because Beaker jobs have no working GCS credential (the workspace
#   GOOGLE_APPLICATION_CREDENTIALS secret is a dead personal token). Weights load
#   via trainer.load_checkpoint with load_trainer_state=False and
#   load_optim_state=False: weight initialization, fresh optimizer, step 0.
# * CONFIG_NAME is a json file, not a preset. The checkpoint's own config.json was
#   written by an older moe-v2-core and needs migration (attention_norm +
#   feed_forward_norm -> layer_norm, drop d_attn, drop ep, flash_4 -> flash_2);
#   scratchpad is untracked, so the migrated config is committed next to this
#   script. Regenerate with scripts/train/debug/migrate_s004_config.py if the
#   olmo-core pin moves.
# * Router lb_loss (0.015) and router z-loss (1e-4) ride inside the model config,
#   so expert-collapse protection matches pretraining without extra plumbing.
# * MAX_SEQ_LENGTH=8192 is s004's native pretraining context (config.json:
#   dataset.sequence_length=8192). No long-context extension exists for s004.
# * LR 2.5e-5 is the Olmo Hybrid SFT rate. It is NOT validated for this
#   architecture: s004 pretrained with Muon at peak LR 4.4e-4, which does not
#   translate to AdamW. If the loss is flat or unstable, this is the first knob.
# * The model trains through the branch's nn.ddp train module, NOT FSDP2:
#   OLMoDDPModel refuses prepare_experts_for_fsdp by design (first smoke run,
#   01M08QTBC0KW7FKVYSJG1H9Y55). The DDP module's optimizer and dp_config come
#   verbatim from the checkpoint's train_module section. Full 26.7B replication
#   per rank: ~53 GB bf16 params + fp32 grads on a 288 GB B300 -- fits, but only
#   because it is Blackwell.
# * No --activation_checkpointing_mode and no --compile_model: at micro-batch
#   1 x 8192 the activations are small next to the replicated param+grad state,
#   and each removed feature is a removed failure mode for a first-of-its-kind
#   run. Pretraining compiled; turn it back on for speed once the path is proven.
# * ai2/holmes (B300, CUDA 13): the FSDP2 reduce-scatter hang killed 4/4 H100
#   runs of the hybrid (allenai/OLMo-core#829), and this model shares the GDN
#   blocks. Do not spend H100 queue time finding out it reproduces.

set -euo pipefail

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
MODE="${2:-train}"

# ---- cache-key arguments: MUST be byte-identical across both jobs ----
MODEL=/weka/oe-adapt-default/abhishekr/s004/step69000
CONFIG_NAME=scripts/train/debug/s004_migrated.json
TOKENIZER=allenai/olmo-3-tokenizer-instruct-dev
CHAT_TEMPLATE=olmo123
MAX_SEQ_LENGTH=8192
MIXER="allenai/Dolci-Instruct-SFT 1.0"
SEED=33333
LOCAL_CACHE_DIR=/weka/oe-adapt-default/allennlp/numpy_sft_cache
# ----------------------------------------------------------------------

# Global batch held at 1,048,576 tokens like every run in this series:
#   1 seq * 16 grad_accum * 8 ranks * 8192 tokens. One epoch of the mixture is
#   ~1723 steps at this batch, so 172 steps is ~0.1 epoch -- where the hybrid had
#   already banked ~80% of its IFBench gain.
#   Smoke-test first: MAX_TRAIN_STEPS=30 CHECKPOINTING_STEPS=30 checks that the
#   26.7B DCP load, a forward/backward, and a checkpoint save all work before
#   committing the node for hours.
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-172}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-86}"
DIST_TIMEOUT_HOURS="${DIST_TIMEOUT_HOURS:-4}"
# Synchronous saves are mandatory here, not a tuning choice: OLMoDDPTrainModule
# raises "does not support async checkpointing" (hit at the step-30 save in
# 01M08XZS450RKD0NHJBSV87MPR, after all 30 steps had trained cleanly).
SAVE_ASYNC_FLAG="--no_save_async"

echo "Using Beaker image: $BEAKER_IMAGE"
echo "Mode: $MODE"

if [[ "$MODE" == "tokenize" ]]; then
    uv run python mason.py \
        --cluster ai2/saturn ai2/neptune ai2/ceres \
        --workspace ai2/open-instruct-dev \
        --priority urgent \
        --image "$BEAKER_IMAGE" \
        --description "Tokenize Dolci-Instruct-SFT (seq 8192, olmo123) for the s004 MoE spike" \
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
        --description "s004 MoE SFT spike, Dolci-Instruct-SFT, $MAX_TRAIN_STEPS steps, 1x8 (seq 8192)" \
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
