#!/bin/bash
# OLMoE3 latent-KDA MoE + Dolci-Think-SFT: the think baseline for future
# iterations. Sibling of oc_sft_olmoe3_kda_1node.sh, which did the same model
# over Dolci-Instruct-SFT. Plan and measurements: ~/handoff/moe-think-plan.md
#
#   ./scripts/train/debug/oc_sft_olmoe3_kda_think.sh $IMAGE $MODE
#
# The launch script runs locally and only ever passes a command line, so a mode
# that changes nothing under open_instruct/ can reuse an existing image rather
# than rebuilding. Everything here is CLI-only by design, for that reason.
#
# Modes:
#   tokenize_subset  0.02 of Dolci-Think at SEQ, CPU-only. Feeds the gate.
#   gate             30 steps, 1x8, at SEQ. Decides whether 16384 fits.
#   discover_cache   CPU-only; fails fast printing the numpy dir the full-think
#                    run expects, so an existing cache can be linked to it.
#   tokenize_full    full Dolci-Think at SEQ, CPU-only. The ~10h long pole.
#   smoke_2node      30 steps, 2x8, on the subset. Tests DDP + 16-rank writes.
#   convert          CPU-only, in-image HF conversion of one checkpoint.
#                    CKPT_ROOT=<dir> STEP=stepNNNN. Must run in-image: the KDA
#                    modules live in olmo-core `akshitab/emo_modularity`, which
#                    the spike branch pins. A local venv synced from main has
#                    NO `olmo_core.nn.attention.kda` and every conversion dies
#                    with ModuleNotFoundError. Uses /stage/.venv/bin/python
#                    directly because `uv run` resyncs and reverts the torch
#                    2.11 companion installs.
#   lr_probe         300 steps, 1x8, on the subset. INSTABILITY SCREEN ONLY --
#                    see the scheduler note below.
#   train            the real run: STEPS steps, 2x8, full think corpus.
#
# Why the settings are what they are (beyond the instruct script's reasons):
#
# * SEQ defaults to 16384, not the checkpoint's native 8192. At 8192, 49.1% of
#   Dolci-Think rows are cut mid-trace; at 16384 it is 30.6%. Truncated rows are
#   unterminated reasoning traces, i.e. supervision to never stop, on a model
#   that already has a non-termination tail. Nothing structural forbids 16384:
#   15/19 blocks are KDA (linear attention, position-free) and the 4
#   full-attention blocks carry no rope config at all, so there is no rope
#   scaling to extend. The open question is memory, hence the gate.
#
# * The gate must be a full-model run, not probe_kda_forward.py. Activation
#   memory here is set by rank_microbatch_size, which open-instruct computes as
#   per_device_train_batch_size * max_seq_length (olmo_core_finetune.py) --
#   the checkpoint config's own rank_microbatch_size is ignored. So 16384
#   doubles the microbatch against the 1-epoch instruct run, from a baseline
#   that already reserved 207 of 268 GiB. olmo-core also requires a microbatch
#   to hold a whole number of sequences, so 16384 is a hard floor at SEQ=16384;
#   per_device_train_batch_size is already 1 and cannot absorb it.
#
# * If the gate OOMs, the fallback ladder is cheapest-first, and the first two
#   rungs need no image rebuild:
#     1. ACT_MEM_BUDGET=0.5 -- budget-mode activation checkpointing. NOTE this
#        is genuinely off today: build_ac_config only returns a budget config
#        when activation_memory_budget < 1.0, and the instruct run passed 1,
#        so it trained with ac_config=None at the train-module level. The
#        block-level checkpoint_attn / checkpoint_permute_moe_unpermute flags
#        in the model json were on; the train-module one was not.
#     2. ACT_CKPT_MODE=selected_modules with the usual module list.
#     3. recompute_each_block / recompute_all_blocks_by_chunk in the model json
#        (both False today) -- this one does need a rebuild.
#
# * Global batch is held at 1,048,576 tokens, as in every run in this series,
#   so runs stay comparable by tokens rather than by steps. GRAD_ACCUM is
#   derived from SEQ and the rank count to keep that invariant.
#
# * lr_probe is a SCREEN, not a selector. build_scheduler derives warmup from
#   num_training_steps and LinearWithWarmup anneals over it, so a 300-step probe
#   warms up in 9 steps and fully anneals inside the window while the real run
#   anneals over thousands. That systematically favours the larger LR and hides
#   the late-run instability that usually disqualifies it. Judge on grad-norm
#   spikes, loss divergence and load-imbalance climb -- not on final CE. To make
#   it select, set STEPS to the anchor length in both arms and stop early so the
#   schedule shape matches.
#
# * KEEP_LAST_N defaults to keeping every permanent checkpoint, because the
#   0.1-epoch grid IS the deliverable. That is deliberate but expensive: each is
#   207 GB, written synchronously. Convert to HF and delete each DCP dir as it
#   lands rather than at the end of the run. Note -1 does NOT mean "keep a few":
#   olmo_core_utils maps it to None and olmo-core then prunes nothing at all.

set -euo pipefail

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
MODE="${2:-gate}"

# ---- cache-key arguments: MUST be byte-identical across tokenize and train ----
# LC arm (#1854): the long-context (65536-native) base, not the midtrain one.
# CONFIG_NAME must be regenerated from THIS checkpoint's config.json -- the
# midtrain-derived kda_mt_sft.json has recompute_each_block=False and an
# unfused loss head, the two levers that make long sequences fit.
# BASE selects the checkpoint + regenerated config pair. Both are overridable
# individually (MODEL, CONFIG_NAME) for a checkpoint that has no preset yet.
#   proxy            the 1.2B/18.5B latent-KDA proxy (~200B PT tokens; #1854..#1880)
#   hero-small-nonemo  Olmo 3.5 hero small, 0.79B/12.5B, 2T PT + 100B MT + 100B LC,
#   hero-small-ptemo   same, but EMO on during PT only (paired arm for the EMO question).
#                    non-EMO throughout (#1895). Lives in olmo-3p5-checkpoints,
#                    which mason does not mount by default -- see EXTRA_BUCKET_FLAGS.
BASE="${BASE:-proxy}"
case "$BASE" in
  proxy)
    DEFAULT_MODEL=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/long_context/legacy-cx8-v2/latent-kda-l2/1.2B/cx8-samebatch/step63802
    DEFAULT_CONFIG=scripts/train/debug/kda_lc_sft.json
    ;;
  hero-small-nonemo)
    DEFAULT_MODEL=/weka/olmo-3p5-checkpoints/production-hero-small-lc/olmo35-small-2t-lc100b-20260913/olmo35-small-2t-lc100b-20260913-non-emo/step5961
    DEFAULT_CONFIG=scripts/train/debug/kda_hero_small_sft.json
    # torch.compile on the packed path gives a non-finite loss at step 1 on this
    # base: the gate died in the optimizer's finite check on both attempts
    # (01M2KV7PJRGQ30KGFS6AZFCS72, "Non-finite loss encountered in
    # OLMoDDPOptimizer"), and Jacob saw the same on both LC sources with the
    # identical eager run finite. Eager until the compiler/varlen interaction is
    # qualified separately. The proxy keeps compiling (its anchors were finite).
    DEFAULT_COMPILE=0
    ;;
  hero-small-ptemo)
    # Same geometry and recipe as hero-small-nonemo; EMO was on during
    # pretraining only (off for MT/LC, so the router config carries no emo
    # field). The paired arm for the EMO-in-PT question; keep every other
    # setting identical to the non-EMO anchor so the comparison stays paired.
    DEFAULT_MODEL=/weka/olmo-3p5-checkpoints/production-hero-small-lc/olmo35-small-2t-lc100b-noemo-20260914/olmo35-small-2t-lc100b-noemo-20260914-emo/step5961
    DEFAULT_CONFIG=scripts/train/debug/kda_hero_small_ptemo_sft.json
    DEFAULT_COMPILE=0
    ;;
  *)
    echo "Unknown BASE=$BASE (expected proxy, hero-small-nonemo or hero-small-ptemo)" >&2
    exit 1
    ;;
esac
MODEL="${MODEL:-$DEFAULT_MODEL}"
CONFIG_NAME="${CONFIG_NAME:-$DEFAULT_CONFIG}"
COMPILE="${COMPILE:-${DEFAULT_COMPILE:-1}}"
if [[ "$COMPILE" == "1" ]]; then COMPILE_FLAG=""; else COMPILE_FLAG="--no_compile_model"; fi
# mason mounts only oe-adapt-default and oe-training-default; a base under any
# other bucket needs --extra_weka_buckets or the job fails at load time with a
# missing path (#1897). Derived from MODEL so a manual override cannot forget it.
EXTRA_BUCKET_FLAGS=""
if [[ "$MODEL" =~ ^/weka/([^/]+)/ ]]; then
    bucket="${BASH_REMATCH[1]}"
    if [[ "$bucket" != "oe-adapt-default" && "$bucket" != "oe-training-default" ]]; then
        EXTRA_BUCKET_FLAGS="--extra_weka_buckets $bucket"
    fi
fi
TOKENIZER=allenai/olmo-3-tokenizer-instruct-dev
CHAT_TEMPLATE=olmo123
# H015: THINK_TOKENS=1 makes <think> and </think> single tokens by renaming reserved
# <|extra_id_N|> slots (#1911). It is part of the tokenization cache key, so every mode
# that names the cache passes it identically. An array, so the tags stay literal words.
RESERVED_SLOT_FLAGS=()
if [[ "${THINK_TOKENS:-0}" == "1" ]]; then
    RESERVED_SLOT_FLAGS=(--reserved_slot_tokens "<think>" "</think>")
fi
# 32768 is the cache's native tokenisation length (no re-tokenize) and cuts
# mid-trace truncation to 1.95%; 16384 would show the LC base the same data as
# the midtrain base.
SEQ="${SEQ:-32768}"
SUBSET_FRAC="${SUBSET_FRAC:-0.02}"
# Probe modes read the full corpus by default. The 0.02 subset existed only to
# avoid a ~10 h tokenize; that is moot now that the 32768 cache is symlinked to
# the 16384 path (see numpy_sft/README-6d4fae1e10-f6bfdc56.md), and probing the
# real cache also validates the symlink. Set PROBE_MIXER to the subset to go
# back. NOTE a "0.02 subset" tokenize job is NOT 2% of the cost: load_dataset
# materialises all 2.25M rows before update_range() samples, so the full-corpus
# load is paid either way.
PROBE_MIXER="${PROBE_MIXER:-allenai/Dolci-Think-SFT 1.0}"
SEED=33333
# Noise-floor seeds change the data order only (--data_loader_seed). SEED is a
# tokenization cache key and must stay fixed (#1880, H001).
DATA_LOADER_SEED="${DATA_LOADER_SEED:-34521}"
# The 32768 Dolci-Think cache from the dense run lives here; keeping think
# caches together is what makes the discover_cache/link shortcut possible.
LOCAL_CACHE_DIR=/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache
# ------------------------------------------------------------------------------

CLUSTER="${CLUSTER:-ai2/holmes}"
WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"
PRIORITY="${PRIORITY:-high}"
LR="${LR:-2.5e-5}"
ACT_MEM_BUDGET="${ACT_MEM_BUDGET:-1}"
ACT_CKPT_MODE="${ACT_CKPT_MODE:-budget}"
# A git worktree has no .venv of its own and `uv run` would sync a fresh one
# (which, per the KDA runbook, also reverts the torch 2.11 companion installs).
# mason.py is pure local orchestration, so any working interpreter will do.
PY="${PY:-uv run python}"
# Holmes uses Strict Priority with Unallocated-Only Backfill (beaker-docs
# compute/clusters.md, scheduling/management.md). Per concept/allocations.md a
# job is "unallocated" when minRuntime is 0/omitted -- which is what mason's
# --preemptible produces -- and unallocated jobs are backfill ONLY. On a full
# holmes that means never scheduling, which is exactly what we saw: repeated
# "67 nodes do not have enough slots available" at priority high.
# PREEMPTIBLE=0 drops the flag so the job is allocated and draws on the
# workspace's holmes allocation instead of scavenging idle capacity.
# An ALLOCATED job on holmes gets an 8 h min-runtime shield by default and, with
# no --timeout, no lifetime cap at all -- so a hang holds 8 B300s and cannot be
# preempted. Always cap probe jobs. mason: "--timeout ... If not specified, no
# timeout is set."
JOB_TIMEOUT="${JOB_TIMEOUT:-45m}"
# mason's --max_retries defaults to 0. Multi-node jobs need EVERY replica up
# within 10 min ("timed out after waiting 10m0s for synchronized replica start"),
# so one bad node kills the whole group -- we lost a 2x8 smoke to a node that got
# cordoned mid-start by an unrelated stuck container. With no retries that is a
# hard failure, which for a multi-hour 2x8 run is a real exposure.
MAX_RETRIES="${MAX_RETRIES:-2}"
# CPU conversion of a 207 GB MoE checkpoint does NOT finish in 2 h -- measured:
# it reached "Loading checkpoint" and was still going 114 min later. CUDA is the
# converter's own default (its KDA/fla Triton kernels reject CPU tensors) and the
# runbook records CPU and CUDA output as bit-identical over 35 GB, so with a GPU
# free there is no reason to take the slow path. CPU remains available for when
# no GPU is.
CONVERT_DEVICE="${CONVERT_DEVICE:-cuda}"
CONVERT_GPUS="${CONVERT_GPUS:-1}"
# The cuda13 image needs a CUDA 13 driver. holmes has it; saturn/ceres/jupiter/
# neptune are on 12.8 and every GPU conversion there dies with "The NVIDIA driver
# on your system is too old (found version 12080)". So a CUDA conversion must run
# on holmes, while a CPU conversion can go anywhere (but takes >2 h).
if [[ "$CONVERT_DEVICE" == "cuda" ]]; then
    CONVERT_CLUSTER="${CONVERT_CLUSTER:-ai2/holmes}"
else
    CONVERT_CLUSTER="${CONVERT_CLUSTER:-ai2/saturn ai2/neptune ai2/ceres ai2/jupiter}"
fi
PREEMPTIBLE="${PREEMPTIBLE:-1}"
if [[ "$PREEMPTIBLE" == "1" ]]; then PREEMPTIBLE_FLAG="--preemptible"; else PREEMPTIBLE_FLAG=""; fi

# Global batch 1,048,576 tokens = SEQ * ranks * GRAD_ACCUM.
grad_accum_for() {
    local ranks=$1
    echo $(( 1048576 / (SEQ * ranks) ))
}

echo "Using Beaker image: $BEAKER_IMAGE"
echo "Mode: $MODE | BASE=$BASE | SEQ=$SEQ | LR=$LR | cluster=$CLUSTER | workspace=$WORKSPACE"
echo "Model: $MODEL"
echo "Config: $CONFIG_NAME ${EXTRA_BUCKET_FLAGS:+| $EXTRA_BUCKET_FLAGS} | compile=$COMPILE"

case "$MODE" in
  tokenize_subset|tokenize_full)
    if [[ "$MODE" == "tokenize_subset" ]]; then
        MIXER="allenai/Dolci-Think-SFT $SUBSET_FRAC"
        DESC="Tokenize Dolci-Think-SFT $SUBSET_FRAC subset (seq $SEQ, olmo123) for the KDA MoE think gate"
    else
        MIXER="allenai/Dolci-Think-SFT 1.0"
        DESC="Tokenize Dolci-Think-SFT full (seq $SEQ, olmo123) for the KDA MoE think baseline"
    fi
    $PY mason.py \
        --cluster ai2/saturn ai2/neptune ai2/ceres ai2/jupiter \
        --workspace "$WORKSPACE" \
        --priority "$PRIORITY" \
        --image "$BEAKER_IMAGE" \
        --description "$DESC" \
        --pure_docker_mode \
        $PREEMPTIBLE_FLAG \
        --num_nodes 1 \
        --gpus 0 \
        --non_resumable \
        --no_auto_dataset_cache \
        ${EXPECTED_NUMPY_CACHE:+--env EXPECTED_NUMPY_CACHE=$EXPECTED_NUMPY_CACHE} \
        $EXTRA_BUCKET_FLAGS \
        -- uv run python open_instruct/olmo_core_finetune.py \
        --model_name_or_path "$MODEL" \
        --config_name $CONFIG_NAME \
        --tokenizer_name_or_path $TOKENIZER \
        --chat_template_name $CHAT_TEMPLATE \
        "${RESERVED_SLOT_FLAGS[@]}" \
        --max_seq_length "$SEQ" \
        --mixer_list $MIXER \
        --local_cache_dir $LOCAL_CACHE_DIR \
        --seed $SEED \
        --cache_dataset_only
    ;;

  discover_cache)
    # The numpy dir is a hash of the tokenization arguments, but the hash is not
    # a function of the declared inputs alone (#1818) -- computing it locally
    # does NOT reproduce what the job computes, verified against the existing
    # 32768 think cache. So ask the job. It raises FileNotFoundError printing
    # the exact path it wants, before touching distributed init or a GPU.
    $PY mason.py \
        --cluster ai2/saturn ai2/neptune ai2/ceres ai2/jupiter \
        --workspace "$WORKSPACE" \
        --priority "$PRIORITY" \
        --image "$BEAKER_IMAGE" \
        --description "Discover the numpy cache dir for full Dolci-Think at seq $SEQ" \
        --pure_docker_mode \
        $PREEMPTIBLE_FLAG \
        --num_nodes 1 \
        --gpus 0 \
        --non_resumable \
        --no_auto_dataset_cache \
        $EXTRA_BUCKET_FLAGS \
        -- uv run python open_instruct/olmo_core_finetune.py \
        --model_name_or_path "$MODEL" \
        --config_name $CONFIG_NAME \
        --tokenizer_name_or_path $TOKENIZER \
        --chat_template_name $CHAT_TEMPLATE \
        "${RESERVED_SLOT_FLAGS[@]}" \
        --max_seq_length "$SEQ" \
        --mixer_list allenai/Dolci-Think-SFT 1.0 \
        --local_cache_dir $LOCAL_CACHE_DIR \
        --seed $SEED \
        --output_dir /tmp/discover
    ;;

  gate|smoke_2node|lr_probe|train)
    STEPS="${STEPS:-30}"
    CKPT_STEPS="${CKPT_STEPS:-1000000}"
    KEEP_LAST_N="${KEEP_LAST_N:-100}"
    case "$MODE" in
      gate)
        NNODES=1; MIXER="$PROBE_MIXER"
        DESC="KDA MoE think GATE: $STEPS steps, 1x8, seq $SEQ (does $SEQ fit?)" ;;
      smoke_2node)
        NNODES=2; MIXER="$PROBE_MIXER"
        DESC="KDA MoE think 2-node smoke: $STEPS steps, 2x8, seq $SEQ" ;;
      lr_probe)
        NNODES=1; MIXER="$PROBE_MIXER"
        STEPS="${STEPS:-300}"
        DESC="KDA MoE think LR screen: lr=$LR, $STEPS steps, 1x8, seq $SEQ" ;;
      train)
        NNODES="${NNODES:-2}"; MIXER="allenai/Dolci-Think-SFT 1.0"
        DESC="KDA MoE LC + Dolci-Think SFT: $STEPS steps, ${NNODES}x8, seq $SEQ, lr=$LR" ;;
    esac
    # NPROC exists for the gate specifically. The gate asks a per-rank memory
    # question, and this is DDP: every rank holds a full replica of the 18.5B
    # params plus its optimizer state, and the microbatch is
    # per_device_train_batch_size * max_seq_length regardless of world size. So
    # one rank answers the same question as eight, and schedules when holmes is
    # full. It does NOT cover DDP comm buffers at 16 ranks -- that is what
    # smoke_2node is for.
    NPROC="${NPROC:-8}"
    RANKS=$(( NNODES * NPROC ))
    GRAD_ACCUM=$(grad_accum_for $RANKS)
    if (( GRAD_ACCUM < 1 )); then
        echo "Error: SEQ=$SEQ x ranks=$RANKS exceeds the 1,048,576-token global batch." >&2
        exit 1
    fi
    # Beaker only sets BEAKER_LEADER_REPLICA_HOSTNAME for multi-node replica
    # sets. Passing --master_addr on a 1-node job yields the endpoint ":29400"
    # and torchrun dies in parse_rendezvous_endpoint before any model is built.
    if (( NNODES > 1 )); then
        RDZV_FLAGS="--node_rank=\$BEAKER_REPLICA_RANK --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME --master_port=29400"
    else
        # Single node still needs an explicit port: torchrun defaults to 29500,
        # and two 1-GPU jobs scheduled onto the SAME host collide with
        # "EADDRINUSE ... port: 29500". That failure happens during rendezvous,
        # before any model is built, so it yields no memory information and is
        # easy to misread as a real result. Randomised per launch.
        RDZV_FLAGS="--master_port=${MASTER_PORT:-$(( 29500 + RANDOM % 400 ))}"
    fi
    echo "ranks=$RANKS grad_accum=$GRAD_ACCUM -> global batch $(( SEQ * RANKS * GRAD_ACCUM )) tokens"
    # Preemption recovery. An allocated holmes job is only shielded for its 8 h
    # minRuntime; after that an urgent job can preempt it (observed 2026-09-03 at
    # exactly 8h00m, #1875, losing 4,770 steps back to the last permanent save).
    # Beaker does not auto-resume it (mason sets autoResume false), so recovery is
    # a manual relaunch with RESUME_FROM=<dir>/stepNNNN. mason mints a fresh
    # CHECKPOINT_OUTPUT_DIR per launch, so the previous run's dir must be named
    # explicitly; olmo-core restores step, optimizer state and data order, and the
    # LR schedule is a pure function of step, so the trajectory is preserved.
    # EPHEMERAL_STEPS bounds the loss: olmo-core keeps only the latest ephemeral
    # checkpoint (207 GB, ~95 s to write from 16 ranks), so 2000 costs ~2% wall
    # and caps a preemption at ~75 min of lost work.
    RESUME_FLAGS="${RESUME_FROM:+--resume_from_checkpoint $RESUME_FROM}"
    # Where the run writes. mason mints CHECKPOINT_OUTPUT_DIR under
    # /weka/oe-adapt-default/allennlp/deletable_checkpoint_states and keeps any
    # explicit --output_dir that already lives under /weka/. That bucket hit 0
    # bytes free on 2026-09-16 (three anchor attempts died in os.makedirs with
    # ENOSPC, 01M2M4N5S1W617B2QKMKBVDPXW), so OUTPUT_DIR lets a run go to another
    # bucket, e.g. /weka/oe-training-default/ai2-llm/checkpoints/abhishekr/<run>.
    # Serve HF exports for evals from oe-adapt-default regardless (olmo-eval's
    # worker-init cap; #1875).
    OUTPUT_DIR_ARG="${OUTPUT_DIR:-\$CHECKPOINT_OUTPUT_DIR}"
    # Probe modes default CKPT_STEPS above STEPS so nothing is written: each
    # checkpoint is 207 GB written synchronously (the DDP train module rejects
    # async), which a memory or LR screen does not need.
    $PY mason.py \
        --cluster "$CLUSTER" \
        --workspace "$WORKSPACE" \
        --priority "$PRIORITY" \
        --image "$BEAKER_IMAGE" \
        --description "$DESC" \
        --pure_docker_mode \
        $PREEMPTIBLE_FLAG \
        --timeout "$JOB_TIMEOUT" \
        --max_retries "$MAX_RETRIES" \
        --num_nodes $NNODES \
        --gpus $NPROC \
        --non_resumable \
        --no_auto_dataset_cache \
        $EXTRA_BUCKET_FLAGS \
        --env OLMO_SHARED_FS=1 \
        ${EXPECTED_NUMPY_CACHE:+--env EXPECTED_NUMPY_CACHE=$EXPECTED_NUMPY_CACHE} \
        -- torchrun \
        --nnodes=$NNODES \
        $RDZV_FLAGS \
        --nproc_per_node=$NPROC \
        open_instruct/olmo_core_finetune.py \
        --model_name_or_path "$MODEL" \
        --config_name $CONFIG_NAME \
        --tokenizer_name_or_path $TOKENIZER \
        --chat_template_name $CHAT_TEMPLATE \
        "${RESERVED_SLOT_FLAGS[@]}" \
        --max_seq_length "$SEQ" \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --learning_rate "$LR" \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --max_grad_norm 1.0 \
        --num_epochs 1 \
        --max_train_steps "$STEPS" \
        --attn_implementation flash_2 \
        $COMPILE_FLAG \
        --activation_checkpointing_mode "$ACT_CKPT_MODE" \
        --activation_memory_budget "$ACT_MEM_BUDGET" \
        --checkpointing_steps "$CKPT_STEPS" \
        --ephemeral_save_interval "${EPHEMERAL_STEPS:--1}" \
        $RESUME_FLAGS \
        --keep_last_n_checkpoints "$KEEP_LAST_N" \
        --dist_timeout_hours "${DIST_TIMEOUT_HOURS:-2}" \
        --no_save_async \
        --with_tracking \
        --run_name "${RUN_NAME:-sft-${BASE}-${MODE}}" \
        --logging_steps 1 \
        --mixer_list $MIXER \
        --local_cache_dir $LOCAL_CACHE_DIR \
        --seed $SEED \
        --data_loader_seed "${DATA_LOADER_SEED:-34521}" \
        --output_dir "$OUTPUT_DIR_ARG"
    ;;

  convert)
    CKPT_ROOT="${CKPT_ROOT:?set CKPT_ROOT to the deletable_checkpoint_states dir}"
    STEP="${STEP:?set STEP, e.g. step1723}"
    # The Olmo 3.5 hero HF export (latent MoE, scalable softmax, per-head QK gains,
    # bundled olmo3moe modeling code) lives on Jacob's olmo-core HF lineage
    # (b1fd2c97, the revision in the base exports' conversion receipts), which
    # diverged from the SFT pin: the pin's exporter raises "The EMo ladder export
    # expects latent_moe=None". CONVERT_PYTHONPATH puts that lineage's src tree
    # (staged on WEKA) ahead of the image's olmo-core for the conversion job only.
    # OLMO_HF_MOE_CORE_REFERENCE / OLMO_USE_TORCH_GROUPED_MM mirror Jacob's
    # conversion environment (reference expert kernels for the verifier).
    CONVERT_ENV_FLAGS="--env OLMO_USE_TORCH_GROUPED_MM=0 --env OLMO_HF_MOE_CORE_REFERENCE=1"
    if [[ -n "${CONVERT_PYTHONPATH:-}" ]]; then
        CONVERT_ENV_FLAGS="$CONVERT_ENV_FLAGS --env PYTHONPATH=$CONVERT_PYTHONPATH"
    fi
    $PY mason.py \
        --cluster $CONVERT_CLUSTER \
        --workspace "$WORKSPACE" \
        --priority "$PRIORITY" \
        --image "$BEAKER_IMAGE" \
        --description "HF-convert KDA MoE think $STEP at seq $SEQ" \
        --pure_docker_mode \
        $PREEMPTIBLE_FLAG \
        --timeout "${JOB_TIMEOUT:-2h}" \
        --num_nodes 1 \
        --gpus "$CONVERT_GPUS" \
        --non_resumable \
        --no_auto_dataset_cache \
        $EXTRA_BUCKET_FLAGS \
        $CONVERT_ENV_FLAGS \
        -- /stage/.venv/bin/python "${CONVERT_SCRIPT:-scripts/train/debug/convert_moe_checkpoint_to_hf.py}" \
        -i "$CKPT_ROOT/$STEP" \
        -o "$CKPT_ROOT/hf_$STEP" \
        -c $CONFIG_NAME \
        -t "${CONVERT_TOKENIZER:-$TOKENIZER}" \
        -s "$SEQ" \
        --skip-validation \
        --device "$CONVERT_DEVICE"
    ;;

  *)
    echo "Unknown mode: $MODE" >&2
    echo "Expected one of: tokenize_subset, tokenize_full, discover_cache, gate, smoke_2node, lr_probe, train, convert" >&2
    exit 1
    ;;
esac
