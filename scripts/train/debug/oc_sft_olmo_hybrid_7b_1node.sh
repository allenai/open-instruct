#!/bin/bash
# Olmo-Hybrid-7B SFT over Dolci-Instruct-SFT, half an epoch, one 8-GPU node.
#
# The procedure -- two-job split, cache key, what a healthy run looks like, the
# eval sweep -- is docs/algorithms/sft_running_guide.md (added in #1835). This
# header records only where the hybrid recipe departs from it.
#
#   ./scripts/train/build_image_and_launch.sh scripts/train/debug/oc_sft_olmo_hybrid_7b_1node.sh tokenize
#   ./scripts/train/build_image_and_launch.sh --cuda-version 13 \
#       scripts/train/debug/oc_sft_olmo_hybrid_7b_1node.sh train
#
# Departures, all forced by the architecture:
#
# * ai2/holmes (B300) and the CUDA 13 image; it hangs on H100 -- see the train branch.
# * --config_name olmo3_hybrid_7B resolves through open_instruct.olmo_core_hybrid,
#   not olmo-core, which has no preset or HF conversion for this architecture.
# * NO --rope_scaling_factor: the model is NoPE. See olmo_core_hybrid.py.
# * LR 2.5e-5, the rate the released Olmo Hybrid SFT used, not Olmo 3's 8e-5.
# * --activation_checkpointing_mode selected_modules: the torch.compile partitioner
#   cannot checkpoint through the opaque GDN `fla` kernels.
#
# Half an epoch, permanent checkpoint every tenth, so the eval curve shows where
# it flattens rather than giving one end-of-run number.

set -euo pipefail

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
MODE="${2:-train}"

# ---- cache-key arguments: MUST be byte-identical across both jobs ----
MODEL=allenai/Olmo-Hybrid-7B
CONFIG_NAME=olmo3_hybrid_7B
# Deliberately the Olmo 3 instruct tokenizer, not the one shipped with
# Olmo-Hybrid-7B: identical BPE, differing only in the names of 10 special tokens
# at ids 100266-100275. Sharing it reuses the cache built for
# oc_sft_olmo3_7b_1node.sh, so the model is the only variable between the runs.
TOKENIZER=allenai/olmo-3-tokenizer-instruct-dev
CHAT_TEMPLATE=olmo123
MAX_SEQ_LENGTH=32768
MIXER="allenai/Dolci-Instruct-SFT 1.0"
SEED=33333
LOCAL_CACHE_DIR=/weka/oe-adapt-default/allennlp/numpy_sft_cache
# ----------------------------------------------------------------------

# Global batch is held at 1,048,576 tokens, matching the Olmo 3 run:
#   1 * 4 * 8 * 32768. One epoch of this mixture is 1723 steps, so half an epoch
#   is 861 and a tenth of an epoch is 172.
#   Override MAX_TRAIN_STEPS for a smoke run, e.g. MAX_TRAIN_STEPS=20 to check
#   that the weights load and the loss starts sane before committing the node.
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-861}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-172}"

# NOTE: a hang here holds its 8 GPUs invisibly for the full 24h distributed
# timeout -- the first attempt at this run went silent at step 580 and sat for
# 2h40m. #1821 exposes --dist_timeout_hours and --save_async to cap that; add
# `--dist_timeout_hours 2` here once it merges.

echo "Using Beaker image: $BEAKER_IMAGE"
echo "Mode: $MODE"

if [[ "$MODE" == "tokenize" ]]; then
    uv run python mason.py \
        --cluster ai2/saturn ai2/neptune ai2/ceres \
        --workspace ai2/open-instruct-dev \
        --priority urgent \
        --image "$BEAKER_IMAGE" \
        --description "Tokenize Dolci-Instruct-SFT for Olmo-Hybrid-7B SFT (seq 32768, olmo123)" \
        --pure_docker_mode \
        --preemptible \
        --num_nodes 1 \
        --gpus 0 \
        --non_resumable \
        --no_auto_dataset_cache \
        -- uv run python open_instruct/olmo_core_finetune.py \
        --model_name_or_path $MODEL \
        --config_name $CONFIG_NAME \
        --tokenizer_name_or_path $TOKENIZER \
        --chat_template_name $CHAT_TEMPLATE \
        --max_seq_length $MAX_SEQ_LENGTH \
        --mixer_list $MIXER \
        --local_cache_dir $LOCAL_CACHE_DIR \
        --seed $SEED \
        --cache_dataset_only
elif [[ "$MODE" == "train" ]]; then
    # Blackwell, not Hopper, and this is not a preference. Four runs of this exact
    # configuration on H100 went silent mid-training -- steps 65, 205, 350 and 580, on four
    # different nodes -- stalled in the FSDP2 post-backward gradient reduce-scatter with
    # every rank enqueued and none completing (allenai/OLMo-core#829). The same
    # configuration completed all 861 steps on a B300 node, and ran 81% faster while doing
    # it (16,392 vs 9,073 tokens/s/device).
    #
    # ai2/holmes requires the CUDA 13 image, hence --cuda-version 13 when building:
    #   ./scripts/train/build_image_and_launch.sh --cuda-version 13 \
    #       scripts/train/debug/oc_sft_olmo_hybrid_7b_1node.sh train
    uv run python mason.py \
        --cluster ai2/holmes \
        --workspace ai2/open-instruct-dev \
        --priority urgent \
        --image "$BEAKER_IMAGE" \
        --description "Olmo-Hybrid-7B SFT, Dolci-Instruct-SFT, 0.5 epoch, 1x8 (seq 32768)" \
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
        --gradient_accumulation_steps 4 \
        --learning_rate 2.5e-5 \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --max_grad_norm 1.0 \
        --num_epochs 1 \
        --max_train_steps $MAX_TRAIN_STEPS \
        --activation_checkpointing_mode selected_modules \
        --attn_implementation flash_2 \
        --compile_model true \
        --checkpointing_steps $CHECKPOINTING_STEPS \
        --ephemeral_save_interval -1 \
        --keep_last_n_checkpoints -1 \
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
