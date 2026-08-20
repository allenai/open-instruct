#!/bin/bash
# Olmo-3-7B SFT over the full Dolci-Instruct-SFT mixture, 1 epoch, one 8-GPU node.
# Derived from oc_sft_olmo3_7b_full.sh (4 nodes, 2 epochs).
#
# Verified end-to-end: https://beaker.org/ex/01KZHT32T30M2VHCWKRJS1G9P7
# 1723 steps, ~7.5 h, exit 0, final CE 0.65.
#
# RUN AS TWO JOBS. Training hard-fails if the pre-tokenized cache is absent, so
# tokenize on CPU first:
#
#   ./scripts/train/build_image_and_launch.sh scripts/train/debug/oc_sft_olmo3_7b_1node.sh tokenize
#   ./scripts/train/build_image_and_launch.sh scripts/train/debug/oc_sft_olmo3_7b_1node.sh train
#
# The cache key hashes the tokenizer config, mixer, transform fns, max_seq_length
# and seed, so both jobs must pass byte-identical values -- hence the shared
# variables below. Any divergence yields a different key and the training job
# fails as though tokenization never ran.
#
# Global batch is held at 1,048,576 tokens, matching the 4-node script:
#   global_batch = per_device(1) * grad_accum * (world_size / cp_degree) * seq_len
#   4 nodes: 1 * 2 * (32/2) * 32768      1 node: 1 * 4 * (8/1) * 32768
#
# Choices that look wrong but are not -- do not "fix" these without re-testing:
#
# * CHAT_TEMPLATE=olmo123 is not a typo. It is unregistered, so it falls through
#   to the tokenizer's own template, which is what the released Olmo 3 Instruct
#   models used. The silent fallback is tracked in #1805.
# * --ephemeral_save_interval -1 must be passed explicitly, not omitted: the
#   parser defaults it to 500, which exceeds save_interval and trips olmo-core's
#   'ephemeral_save_interval must be less than save_interval' check. Together
#   with --keep_last_n_checkpoints -1 this stops olmo-core deleting old
#   checkpoints; deleting a ~100 GB tree on Weka overruns a 30 s timeout and
#   kills the job.
# * No --cp_degree / --cp_strategy. ulysses is incompatible with
#   --rope_scaling_factor, and every ring strategy needs ring-flash-attn, which
#   is not in this image. grad_accum 8 -> 4 compensates.
# * --activation_checkpointing_mode selected_modules rather than
#   --activation_memory_budget: without CP each rank holds a full 32768-token
#   sequence and budget mode OOMs. Budget mode is also a silent no-op unless
#   --compile_model is true.
# * No torchrun --node_rank / --master_addr / --master_port. Beaker only sets the
#   replica env vars for multi-replica jobs, so at --num_nodes 1 they expand to
#   empty and torchrun dies. Single-node torchrun defaults to localhost.

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
LOCAL_CACHE_DIR=/weka/oe-adapt-default/allennlp/numpy_sft_cache
# add_bos must stay off: dataset_transformation.py asserts it for any "olmo*"
# chat template.
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
    # Scheduling an 8-GPU slot on jupiter is the hard part. Run `beaker job
    # events <job-id>` on anything that stays queued -- the scheduler prints its
    # own reason, and the two reasons need opposite fixes:
    #
    #   "workspace X using 64/64 allowed unallocated slots"
    #       The cap is per workspace and applies to every cluster at once, so
    #       switching cluster does nothing; switching workspace does. Seen on
    #       01KZFDBP3GPQ82VH7JAK2TWF2B for 1h32m.
    #   "N nodes do not have enough slots available"
    #       Genuine capacity. Only priority or patience helps.
    #
    # ai2/abhishekr clears the first but caps max workload priority at "normal"
    # (urgent 400s at experiment-create), and a normal-priority job on jupiter's
    # strict-priority scheduler only gets backfill: 01KZFK3MPNJKDEYH4GC0VWKC85
    # starved there for 15h29m without ever scheduling. open-instruct-dev with
    # urgent is the combination that actually runs, whenever its cap allows.
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
        --nproc_per_node=8 \
        open_instruct/olmo_core_finetune.py \
        --model_name_or_path $MODEL \
        --config_name olmo3_7B \
        --tokenizer_name_or_path $TOKENIZER \
        --chat_template_name $CHAT_TEMPLATE \
        --max_seq_length $MAX_SEQ_LENGTH \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 8e-5 \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --max_grad_norm 1.0 \
        --num_epochs 1 \
        --rope_scaling_factor 8 \
        --activation_checkpointing_mode selected_modules \
        --attn_implementation flash_2 \
        --compile_model true \
        --checkpointing_steps 345 \
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
