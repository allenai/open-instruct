#!/bin/bash
# Olmo-3-7B SFT on the full Dolci-Instruct-SFT mixture, 1 epoch, one 8-GPU node.
#
# Derived from oc_sft_olmo3_7b_full.sh (4 nodes, 2 epochs). Two changes keep the
# global batch identical at 1,048,576 tokens:
#
#   global_batch = per_device(1) * grad_accum * (world_size / cp_degree) * seq_len
#
# 4 nodes: 1 * 2 * (32/2) * 32768 = 1048576
# 1 node:  1 * 8 * (8/2)  * 32768 = 1048576
#
# so grad_accum goes 2 -> 8, and num_epochs 2 -> 1. cp_degree, cp_strategy,
# activation_memory_budget and attn_implementation are unchanged from the 4-node
# script.
#
# Expect ~1316 steps: Dolci-Instruct-SFT is ~1.38B tokens after tokenization
# (2,124,202 sequences, measured in https://github.com/allenai/open-instruct/pull/1806),
# and 1.38e9 / 1048576 = 1316. If the log's "Total training steps" is far from
# that, the batch math is wrong -- stop and check before burning the budget.
#
# `olmo123` is deliberate and is NOT a typo. It is an unregistered name, so
# get_tokenizer_tulu_v2_2 falls through to the tokenizer's own chat template
# (dataset_transformation.py:838-846). That is the template the released Olmo 3
# Instruct models were built with. See issue #1805 -- the silent fallback is a
# known wart, not an accident here.
#
# TWO JOBS. Training hard-fails if the pre-tokenized numpy cache is absent
# (olmo_core_finetune.py:149). Tokenize first on CPU, then train:
#
#   ./scripts/train/build_image_and_launch.sh scripts/train/debug/oc_sft_olmo3_7b_1node.sh tokenize
#   ./scripts/train/build_image_and_launch.sh scripts/train/debug/oc_sft_olmo3_7b_1node.sh train
#
# Both modes read the cache-key arguments from the same variables below. Do not
# inline them per-mode: the cache key is a hash of the tokenizer config, the
# mixer, the transform fns, max_seq_length and the seed
# (olmo_core_finetune.py:135-137), so any divergence between the two jobs
# silently produces a different key and the training job then fails as if
# tokenization had never run.

set -euo pipefail

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
MODE="${2:-train}"

# ---- cache-key arguments: MUST be byte-identical across both jobs ----
MODEL=allenai/Olmo-3-1025-7B
TOKENIZER=allenai/olmo-3-tokenizer-instruct-dev
CHAT_TEMPLATE=olmo123
MAX_SEQ_LENGTH=32768
MIXER="allenai/Dolci-Instruct-SFT 1.0"
SEED=33333
# Passed for parity with the 4-node script, but note it does not take effect on
# Beaker: olmo_core_utils.py:467-468 overrides local_cache_dir to
# /weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache for any
# Beaker job. Both modes are Beaker jobs, so both land on the same Weka path.
LOCAL_CACHE_DIR=/weka/oe-adapt-default/allennlp/numpy_sft_cache
# add_bos, mixer_list_splits and transform_fn are left at their defaults in both
# modes, which is what keeps them out of the way here. add_bos in particular must
# stay off: dataset_transformation.py:789 asserts it for any "olmo*" template.
# ----------------------------------------------------------------------

echo "Using Beaker image: $BEAKER_IMAGE"
echo "Mode: $MODE"

if [[ "$MODE" == "tokenize" ]]; then
    # CPU-only, and NOT on jupiter: an 8-GPU slot there queues for hours while a
    # 0-GPU slot on saturn/neptune/ceres schedules in minutes. The cache lands on
    # Weka, which the training job reads from.
    uv run python mason.py \
        --cluster ai2/saturn ai2/neptune ai2/ceres \
        --workspace ai2/open-instruct-dev \
        --priority urgent \
        --image "$BEAKER_IMAGE" \
        --description "Tokenize Dolci-Instruct-SFT for Olmo-3-7B SFT (seq 32768, olmo123)" \
        --pure_docker_mode \
        --preemptible \
        --num_nodes 1 \
        --gpus 0 \
        --non_resumable \
        --no_auto_dataset_cache \
        -- uv run python open_instruct/olmo_core_finetune.py \
        --model_name_or_path $MODEL \
        --tokenizer_name_or_path $TOKENIZER \
        --chat_template_name $CHAT_TEMPLATE \
        --max_seq_length $MAX_SEQ_LENGTH \
        --mixer_list $MIXER \
        --local_cache_dir $LOCAL_CACHE_DIR \
        --seed $SEED \
        --cache_dataset_only
elif [[ "$MODE" == "train" ]]; then
    uv run python mason.py \
        --cluster ai2/jupiter \
        --workspace ai2/open-instruct-dev \
        --priority urgent \
        --image "$BEAKER_IMAGE" \
        --description "Olmo-3-7B SFT, Dolci-Instruct-SFT, 1 epoch, 1x8 (seq 32768)" \
        --pure_docker_mode \
        --preemptible \
        --num_nodes 1 \
        --gpus 8 \
        --non_resumable \
        --no_auto_dataset_cache \
        --env OLMO_SHARED_FS=1 \
        -- torchrun \
        --nnodes=1 \
        --node_rank=\$BEAKER_REPLICA_RANK \
        --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME \
        --master_port=29400 \
        --nproc_per_node=8 \
        open_instruct/olmo_core_finetune.py \
        --model_name_or_path $MODEL \
        --config_name olmo3_7B \
        --tokenizer_name_or_path $TOKENIZER \
        --chat_template_name $CHAT_TEMPLATE \
        --max_seq_length $MAX_SEQ_LENGTH \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 8e-5 \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --max_grad_norm 1.0 \
        --num_epochs 1 \
        --rope_scaling_factor 8 \
        --activation_memory_budget 0.5 \
        --cp_degree 2 \
        --cp_strategy ulysses \
        --attn_implementation flash_2 \
        --compile_model true \
        --checkpointing_steps 1000 \
        --ephemeral_save_interval 200 \
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
