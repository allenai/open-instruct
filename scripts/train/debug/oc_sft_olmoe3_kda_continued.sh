#!/bin/bash
# Continued SFT on the OLMoE3 latent-KDA 1.2B MoE, starting from the 1-epoch
# Dolci-Think checkpoint of https://github.com/allenai/open-instruct/issues/1859,
# for a second stage on agentic data. Derived from oc_sft_olmoe3_kda_1node.sh
# (the tracked ancestor of the run that produced that checkpoint) with the
# recipe #1859 actually ran: seq 65536, 2x8 B300, LR 2.5e-5, kda_lc_sft.json.
#
#   MIXER="<hf-dataset> 1.0" ./scripts/train/build_image_and_launch.sh --cuda-version 13 \
#       scripts/train/debug/oc_sft_olmoe3_kda_continued.sh tokenize
#   MIXER="<hf-dataset> 1.0" ./scripts/train/build_image_and_launch.sh --cuda-version 13 \
#       scripts/train/debug/oc_sft_olmoe3_kda_continued.sh smoke
#   MIXER="<hf-dataset> 1.0" ./scripts/train/build_image_and_launch.sh --cuda-version 13 \
#       scripts/train/debug/oc_sft_olmoe3_kda_continued.sh train
#
# MIXER has no default on purpose: the agentic mixture is the one thing this run
# exists to vary, and a silent fallback to Dolci-Think would train another epoch
# of the data the base already saw. Everything else has a default and an env
# override.
#
# Run the three modes in order. Training hard-fails if the pre-tokenized cache is
# absent, and the cache key hashes the tokenizer, chat template, mixer, transform
# fns, max_seq_length and seed, so those must be byte-identical across the jobs.
# The smoke (30 steps, no checkpoint) is the memory gate: #1855 measured 190 GiB
# reserved at 65536 with documents <= 32768 and explicitly left memory at
# genuinely-65536-token documents untested. Agentic trajectories are the first
# data that will hit that, so gate it rather than reasoning about it.
#
# Why the settings are what they are:
#
# * MODEL is the olmo-core DCP directory, not the hf_step23607 export next to it.
#   open-instruct loads olmo-core checkpoints weights-only (fresh optimizer,
#   step 0), which is what continued SFT wants. The HF export exists for eval
#   and has no loader on this path.
# * CONFIG_NAME is kda_lc_sft.json, the config #1859 trained with. It carries
#   recompute_each_block and lm_head fused_linear, which are what keep 65536 in
#   memory. Never substitute kda_mt_sft.json (both off; an OOM there would be
#   misdiagnosed as "64K doesn't fit"). The DDP train module reads the optimizer
#   and dp_config from this file verbatim.
# * Tokenizer and chat template are held identical to the SFT that produced the
#   base. --chat_template_name olmo123 is not in CHAT_TEMPLATES, so it falls
#   through to the tokenizer repo's own template (#1805); that is the template
#   the base trained on. Changing the prompt format at this boundary would
#   confound the data change with a template change.
# * MAX_SEQ_LENGTH 65536 is the base's native window and the length #1859
#   trained at. Training this family below its window destroyed extrapolation
#   past the window (RULER at 65536: 0.0883 for a 32768-trained checkpoint vs
#   0.40-0.61 for 65536-trained ones, #1854 / #1859). Agentic trajectories are
#   long, so do not drop this to make a smaller node count fit.
# * Global batch is held at 1,048,576 tokens like every run in this series, so
#   GRAD_ACCUM is derived: 1048576 / (SEQ * NNODES * NPROC). At 65536 that is 1
#   on 2x8 and 2 on 1x8; 16 ranks is the maximum. World size, not sequence
#   length, is what moves throughput (~10% per token from 16 vs 32 ranks,
#   #1855), so 2x8 is the default.
# * LR 2.5e-5 with warmup 0.03 is #1859's rate, inherited from the Olmo Hybrid
#   SFT recipe and never swept for this architecture (#1853 lists it as the
#   fourth search axis). Not lowered for the second stage: this is an untested
#   guess either way and matching the first stage keeps one variable.
# * No --activation_checkpointing_mode and no --rope_scaling_factor. The model
#   is NoPE, and the checkpointing this architecture needs (attn + MoE
#   permute/unpermute) is inside the config json, not an open-instruct flag.
# * --no_save_async is mandatory: OLMoDDPTrainModule raises on async
#   checkpointing. Each checkpoint is ~207 GB and written synchronously, so keep
#   CHECKPOINTING_STEPS coarse and budget the storage (#1853: ~2 TB per long
#   run). --keep_last_n_checkpoints -1 stops olmo-core deleting old ones, since
#   deleting a tree that size on Weka overruns a 30 s timeout and kills the job.
# * No --preemptible on holmes. It sets minRuntime 0, which makes the job
#   backfill-only on a strict-priority cluster, so it never schedules when full
#   (70 min of FailedScheduling vs 30 s without it, #1853). An allocated job gets
#   an 8 h min-runtime shield instead, hence an explicit --timeout.
# * ai2/holmes (B300) and the CUDA 13 image are requirements, not preferences:
#   the full 18.5B parameters are replicated per rank (no expert parallelism),
#   which does not fit H100/H200, and torch 2.10 breaks every KDA kernel.
#
# Known gaps to weigh before trusting the result:
#
# * Rows longer than MAX_SEQ_LENGTH are head-truncated at tokenize time and lose
#   their EOS, which trains the model to never stop (#1859 thread, fixed on main
#   by --over_length_strategy in #1876 but not yet on this branch). Measure the
#   trajectory length distribution first: at 65536 Dolci-Think had one such
#   row, but agentic data may have many. If it does, merge #1876 before
#   tokenizing rather than training on it.
# * sft_tulu_filter_v1 dropped 82% of Dolci Instruct Tool Use rows as
#   prefix-unstable (#1853). Read the tokenize job's row counts before and after
#   the filter; a tool-bearing mixture that is silently 18% of its nominal size
#   invalidates the run.
# * Termination rate is this family's unmeasured weakness (~41% of open-ended
#   prompts run to the token cap after Dolci-Think SFT). Cap max_tokens on every
#   eval and report the non-termination rate alongside accuracy.

set -euo pipefail

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
MODE="${2:-smoke}"

# ---- cache-key arguments: MUST be byte-identical across all modes ----
if [[ -z "${MIXER:-}" ]]; then
    echo "MIXER is required, e.g. MIXER=\"allenai/<agentic-sft-dataset> 1.0\" (space-separated dataset/weight pairs)" >&2
    exit 1
fi
TOKENIZER="${TOKENIZER:-allenai/olmo-3-tokenizer-instruct-dev}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-olmo123}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-65536}"
SEED="${SEED:-33333}"
LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/weka/oe-adapt-default/allennlp/numpy_sft_cache}"
# ----------------------------------------------------------------------

# The durable copy of the #1859 1-epoch checkpoint. The scratch copy under
# deletable_checkpoint_states/ccta7vt6/ is expendable; do not point here at it.
MODEL="${MODEL:-/weka/oe-adapt-default/abhishekr/checkpoints/olmoe3-kda-1.2b-dolci-think-sft-65536/step23607}"
CONFIG_NAME="${CONFIG_NAME:-scripts/train/debug/kda_lc_sft.json}"

LR="${LR:-2.5e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
NNODES="${NNODES:-2}"
NPROC="${NPROC:-8}"
GLOBAL_BATCH_TOKENS="${GLOBAL_BATCH_TOKENS:-1048576}"
if (( GLOBAL_BATCH_TOKENS % (MAX_SEQ_LENGTH * NNODES * NPROC) != 0 )); then
    echo "GLOBAL_BATCH_TOKENS=$GLOBAL_BATCH_TOKENS is not a multiple of SEQ*NNODES*NPROC=$(( MAX_SEQ_LENGTH * NNODES * NPROC ))" >&2
    exit 1
fi
GRAD_ACCUM="${GRAD_ACCUM:-$(( GLOBAL_BATCH_TOKENS / (MAX_SEQ_LENGTH * NNODES * NPROC) ))}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}"
DIST_TIMEOUT_HOURS="${DIST_TIMEOUT_HOURS:-4}"

CLUSTER="${CLUSTER:-ai2/holmes}"
WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"
PRIORITY="${PRIORITY:-urgent}"
MAX_RETRIES="${MAX_RETRIES:-0}"
PY="${PY:-uv run python}"

echo "Using Beaker image: $BEAKER_IMAGE"
echo "Mode: $MODE"
echo "Mixer: $MIXER  seq: $MAX_SEQ_LENGTH  nodes x gpus: ${NNODES}x${NPROC}  grad_accum: $GRAD_ACCUM"

# Arguments shared by every mode, so the cache the tokenize job writes hashes
# identically to the one training looks for.
common_args=(
    --model_name_or_path "$MODEL"
    --config_name "$CONFIG_NAME"
    --tokenizer_name_or_path "$TOKENIZER"
    --chat_template_name "$CHAT_TEMPLATE"
    --max_seq_length "$MAX_SEQ_LENGTH"
    --mixer_list $MIXER
    --local_cache_dir "$LOCAL_CACHE_DIR"
    --seed "$SEED"
)

case "$MODE" in
    tokenize)
        # CPU-only, and NOT on holmes or jupiter: a 0-GPU slot on saturn/neptune/
        # ceres schedules in minutes. The cache lands on Weka, which training reads.
        $PY mason.py \
            --cluster ai2/saturn ai2/neptune ai2/ceres \
            --workspace "$WORKSPACE" \
            --priority "$PRIORITY" \
            --image "$BEAKER_IMAGE" \
            --description "Tokenize $MIXER for OLMoE3 KDA continued SFT (seq $MAX_SEQ_LENGTH, $CHAT_TEMPLATE)" \
            --pure_docker_mode \
            --timeout "${TOKENIZE_TIMEOUT:-12h}" \
            --num_nodes 1 \
            --gpus 0 \
            --non_resumable \
            --no_auto_dataset_cache \
            -- uv run python open_instruct/olmo_core_finetune.py \
            "${common_args[@]}" \
            --cache_dataset_only
        ;;
    smoke|train)
        case "$MODE" in
            # No checkpoint: a memory gate does not need a 207 GB synchronous write.
            # --checkpointing_steps must still exceed the step cap or olmo-core saves
            # at the end anyway.
            smoke)
                STEPS="${STEPS:-30}"
                EXTRA=(--max_train_steps "$STEPS" --checkpointing_steps $(( STEPS + 1 )))
                JOB_TIMEOUT="${JOB_TIMEOUT:-2h}"
                DESC="OLMoE3 KDA continued SFT smoke: $STEPS steps, ${NNODES}x${NPROC} (seq $MAX_SEQ_LENGTH)"
                ;;
            train)
                EXTRA=(--checkpointing_steps "$CHECKPOINTING_STEPS" --with_tracking)
                if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
                    EXTRA+=(--max_train_steps "$MAX_TRAIN_STEPS")
                fi
                # The 1-epoch Dolci-Think run took 14.9 h at this configuration for
                # 24.75B tokens (#1859). Size this to the mixture, with headroom.
                JOB_TIMEOUT="${JOB_TIMEOUT:-24h}"
                DESC="OLMoE3 KDA continued SFT: $MIXER, $NUM_EPOCHS epoch, ${NNODES}x${NPROC} (seq $MAX_SEQ_LENGTH)"
                ;;
        esac
        if (( NNODES > 1 )); then
            RDZV_FLAGS="--node_rank=\$BEAKER_REPLICA_RANK --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME --master_port=29400"
        else
            # Beaker only sets the replica env vars for multi-replica jobs; at one
            # node they expand to empty and torchrun dies. Randomize the port so two
            # single-node jobs on one host do not collide during rendezvous.
            RDZV_FLAGS="--master_port=${MASTER_PORT:-$(( 29500 + RANDOM % 400 ))}"
        fi
        $PY mason.py \
            --cluster "$CLUSTER" \
            --workspace "$WORKSPACE" \
            --priority "$PRIORITY" \
            --image "$BEAKER_IMAGE" \
            --description "$DESC" \
            --pure_docker_mode \
            --timeout "$JOB_TIMEOUT" \
            --max_retries "$MAX_RETRIES" \
            --num_nodes "$NNODES" \
            --gpus "$NPROC" \
            --non_resumable \
            --no_auto_dataset_cache \
            --env OLMO_SHARED_FS=1 \
            -- torchrun --nnodes="$NNODES" $RDZV_FLAGS --nproc_per_node="$NPROC" \
            open_instruct/olmo_core_finetune.py \
            "${common_args[@]}" \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps "$GRAD_ACCUM" \
            --learning_rate "$LR" \
            --warmup_ratio "$WARMUP_RATIO" \
            --weight_decay 0.0 \
            --max_grad_norm 1.0 \
            --num_epochs "$NUM_EPOCHS" \
            --attn_implementation flash_2 \
            --ephemeral_save_interval -1 \
            --keep_last_n_checkpoints -1 \
            --dist_timeout_hours "$DIST_TIMEOUT_HOURS" \
            --no_save_async \
            --logging_steps 1 \
            --data_loader_seed 34521 \
            --output_dir \$CHECKPOINT_OUTPUT_DIR \
            "${EXTRA[@]}"
        ;;
    *)
        echo "Unknown mode: $MODE (expected 'tokenize', 'smoke' or 'train')" >&2
        exit 1
        ;;
esac
