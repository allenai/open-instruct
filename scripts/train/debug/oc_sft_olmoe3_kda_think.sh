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
MODEL=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/midtraining/mt-1p2b-kda-ev2-neg-nope-gated-latentmoe-l2-paper-cx8-samebatch-lr1p6e-4-r1/step63802
CONFIG_NAME=scripts/train/debug/kda_mt_sft.json
TOKENIZER=allenai/olmo-3-tokenizer-instruct-dev
CHAT_TEMPLATE=olmo123
SEQ="${SEQ:-16384}"
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

# Global batch 1,048,576 tokens = SEQ * ranks * GRAD_ACCUM.
grad_accum_for() {
    local ranks=$1
    echo $(( 1048576 / (SEQ * ranks) ))
}

echo "Using Beaker image: $BEAKER_IMAGE"
echo "Mode: $MODE | SEQ=$SEQ | LR=$LR | cluster=$CLUSTER | workspace=$WORKSPACE"

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
        --preemptible \
        --num_nodes 1 \
        --gpus 0 \
        --non_resumable \
        --no_auto_dataset_cache \
        -- uv run python open_instruct/olmo_core_finetune.py \
        --model_name_or_path "$MODEL" \
        --config_name $CONFIG_NAME \
        --tokenizer_name_or_path $TOKENIZER \
        --chat_template_name $CHAT_TEMPLATE \
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
        --preemptible \
        --num_nodes 1 \
        --gpus 0 \
        --non_resumable \
        --no_auto_dataset_cache \
        -- uv run python open_instruct/olmo_core_finetune.py \
        --model_name_or_path "$MODEL" \
        --config_name $CONFIG_NAME \
        --tokenizer_name_or_path $TOKENIZER \
        --chat_template_name $CHAT_TEMPLATE \
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
        NNODES=2; MIXER="allenai/Dolci-Think-SFT 1.0"
        DESC="KDA MoE + Dolci-Think SFT: $STEPS steps, 2x8, seq $SEQ, lr=$LR" ;;
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
        RDZV_FLAGS=""
    fi
    echo "ranks=$RANKS grad_accum=$GRAD_ACCUM -> global batch $(( SEQ * RANKS * GRAD_ACCUM )) tokens"
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
        --preemptible \
        --num_nodes $NNODES \
        --gpus $NPROC \
        --non_resumable \
        --no_auto_dataset_cache \
        --env OLMO_SHARED_FS=1 \
        -- torchrun \
        --nnodes=$NNODES \
        $RDZV_FLAGS \
        --nproc_per_node=$NPROC \
        open_instruct/olmo_core_finetune.py \
        --model_name_or_path "$MODEL" \
        --config_name $CONFIG_NAME \
        --tokenizer_name_or_path $TOKENIZER \
        --chat_template_name $CHAT_TEMPLATE \
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
        --activation_checkpointing_mode "$ACT_CKPT_MODE" \
        --activation_memory_budget "$ACT_MEM_BUDGET" \
        --checkpointing_steps "$CKPT_STEPS" \
        --ephemeral_save_interval -1 \
        --keep_last_n_checkpoints "$KEEP_LAST_N" \
        --dist_timeout_hours "${DIST_TIMEOUT_HOURS:-2}" \
        --no_save_async \
        --with_tracking \
        --logging_steps 1 \
        --mixer_list $MIXER \
        --local_cache_dir $LOCAL_CACHE_DIR \
        --seed $SEED \
        --data_loader_seed 34521 \
        --output_dir \$CHECKPOINT_OUTPUT_DIR
    ;;

  *)
    echo "Unknown mode: $MODE" >&2
    echo "Expected one of: tokenize_subset, tokenize_full, discover_cache, gate, smoke_2node, lr_probe, train" >&2
    exit 1
    ;;
esac
