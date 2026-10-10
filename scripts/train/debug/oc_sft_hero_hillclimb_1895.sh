#!/bin/bash
# H010 matched tokenizer ablation. Invoke through build_image_and_launch.sh.
# The legacy image is immutable and is the successful H008 anchor's image.
# Core code, lockfile and model config are unchanged between that image's
# 1d8658f87 and the b1714871b parent of this branch. Only aligned uses the fix.
set -euo pipefail
BUILT_IMAGE="${1:?built image required}"
ARM="${2:?aligned or legacy required}"
MODE="${3:-train}"
# H008 anchor-lr5e-5 uses 34521; its seed2 companion uses 34522. SEED (33333) is
# a tokenization cache key and must never move -- vary the data order only.
# Set before the case block: train_full puts the seed in the output dir.
DATA_LOADER_SEED="${DATA_LOADER_SEED:-34521}"
# EXPERIMENT=<hNNN> (anything but h010) gives each arm, mode and data seed its own run
# name, output dir and convert source, prefixed by the tag; the h010 defaults put control and seed2 in one existing dir.
EXPERIMENT="${EXPERIMENT:-h010}"
BUDGET="${BUDGET:-}"
H015_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-hillclimb-1895
# Local, never inherited: an exported RUN_TAG must not rename another mode.
RUN_TAG=""
# The arm alone decides THINK_TOKENS (a tokenization cache key): an inherited value must
# never turn an aligned control into a think run. EXPECTED_NUMPY_CACHE makes a job built
# from this branch die before tokenizing if its arguments resolve to a different cache than
# the arm's; the immutable legacy image predates the check and ignores it.
case "$ARM" in
    aligned) IMAGE="$BUILT_IMAGE"; THINK_TOKENS=0; ARM_CACHE=062b8a3d20-6068a350 ;;
    legacy) IMAGE=01M2KSSB3FCYJ8PNB7B672N9SP; THINK_TOKENS=0; ARM_CACHE=15bfc110a1-6068a350 ;;
    # H015: aligned plus single-token <think>/</think> in reserved slots (#1911). Its cache
    # hash is known only once the tokenize job has run; pass it as THINK_CACHE.
    think)
        IMAGE="$BUILT_IMAGE"; THINK_TOKENS=1; ARM_CACHE="${THINK_CACHE:-}"
        case "$ARM_CACHE" in
            062b8a3d20-6068a350|15bfc110a1-6068a350)
                echo "THINK_CACHE=$ARM_CACHE is a flag-off cache; the think arm needs its own" >&2; exit 1 ;;
        esac
        ;;
    # H038: Dolci-Think on the Olmo 3.5 tokenizer, alone (the control) or with tool-use sets.
    # THINK_TOKENS stays 0: the olmo35 template emits <think> from reasoning_content and the
    # tags are ordinary BPE pieces, as in the H028 anchors. The cache hash is known only once
    # the tokenize job has run; pass it as H038_CACHE. Steps come from the cache token count
    # (0.5 epoch of the union), passed as TRAIN_FULL_STEPS.
    # H054: continued SFT from an SFT checkpoint (MODEL=<run dir>/stepN) on one Tmax terminal-agent
    # set alone, converted to the open-instruct layout (ledger experiments/h054/convert_tmax_oi.py).
    # Same tokenizer, template and cache machinery as the H038/H049 arms; LR comes from the caller
    # (the Tmax SFT recipe is 2e-5, 2 epochs; steps = ceil(2 x cache tokens / 1,048,576) via
    # TRAIN_FULL_STEPS). Run with ROW_ALIGNED_PARTS=1 DOCB=1 as H049's nemo-docb.
    olmo35|olmo35-simfc|olmo35-nemotron|olmo35-both|olmo35-tmax-sft|olmo35-tmax-glm52|olmo35-tmax-big)
        IMAGE="$BUILT_IMAGE"; THINK_TOKENS=0; ARM_CACHE="${H038_CACHE:-}"
        export TOKENIZER=allenai/dolma2-tokenizer-olmo35
        export TOKENIZER_REVISION=8b9717061fae09d5be814373d189919a62a9a00d
        export CHAT_TEMPLATE=olmo35
        case "$ARM" in
            olmo35) FULL_MIXER="allenai/Dolci-Think-SFT 1.0" ;;
            olmo35-simfc) FULL_MIXER="allenai/Dolci-Think-SFT 1.0 allenai/simfc-thinking-qwen35 1.0" ;;
            olmo35-nemotron) FULL_MIXER="allenai/Dolci-Think-SFT 1.0 allenai/nemotron-sft-agentic-v2-tool-calling-oi 1.0" ;;
            olmo35-both) FULL_MIXER="allenai/Dolci-Think-SFT 1.0 allenai/simfc-thinking-qwen35 1.0 allenai/nemotron-sft-agentic-v2-tool-calling-oi 1.0" ;;
            olmo35-tmax-sft) FULL_MIXER="allenai/tmax-sft-oi 1.0" ;;
            olmo35-tmax-glm52) FULL_MIXER="allenai/tmax-sft-glm-52-oi 1.0" ;;
            olmo35-tmax-big) FULL_MIXER="allenai/tmax-sft-big-oi 1.0" ;;
        esac
        export FULL_MIXER
        export PROBE_MIXER="$FULL_MIXER"
        case "${ARM_CACHE%-rowaligned}" in
            062b8a3d20-6068a350|15bfc110a1-6068a350|3f12323c3b-6068a350)
                echo "H038_CACHE=$ARM_CACHE is an olmo123 cache; the olmo35 arms need their own" >&2; exit 1 ;;
        esac
        ;;
    *) echo "Unknown arm: $ARM (expected aligned, legacy, think or olmo35[-simfc|-nemotron|-both|-tmax-sft|-tmax-glm52|-tmax-big])" >&2; exit 1 ;;
esac
# H049: ROW_ALIGNED_PARTS=1 tokenizes/trains on the row-aligned layout ("-rowaligned" cache),
# DOCB=1 takes document boundaries from its per-row metadata instead of EOS (needs
# ROW_ALIGNED_PARTS=1). Both default off, which leaves every existing arm unchanged.
# The cache name must agree with the layout, so a row-aligned run can never read a
# mid-row-cut cache or the reverse. Validated again, and turned into flags, downstream.
ROW_ALIGNED_PARTS="${ROW_ALIGNED_PARTS:-0}"
DOCB="${DOCB:-0}"
BOUNDARY_TAG=""
if [[ "$ROW_ALIGNED_PARTS" == "1" || "$DOCB" == "1" ]]; then
    if [[ "$ARM" != olmo35* ]]; then
        echo "ROW_ALIGNED_PARTS/DOCB apply to the olmo35 arms only (the others pin pre-#843 images or caches)" >&2; exit 1
    fi
    BOUNDARY_TAG=$([[ "$DOCB" == "1" ]] && echo "-docb" || echo "-rowaligned")
fi
if [[ -n "$ARM_CACHE" ]]; then
    if [[ "$ROW_ALIGNED_PARTS" == "1" && "$ARM_CACHE" != *-rowaligned ]]; then
        echo "ROW_ALIGNED_PARTS=1 needs a -rowaligned cache; H038_CACHE=$ARM_CACHE is mid-row-cut" >&2; exit 1
    fi
    if [[ "$ROW_ALIGNED_PARTS" != "1" && "$ARM_CACHE" == *-rowaligned ]]; then
        echo "H038_CACHE=$ARM_CACHE is row-aligned; set ROW_ALIGNED_PARTS=1 to use it" >&2; exit 1
    fi
fi
export ROW_ALIGNED_PARTS DOCB
case "$MODE" in
    train)
        export STEPS=3072 NNODES=2 NPROC=8 CKPT_STEPS=3072 EPHEMERAL_STEPS=1024
        export JOB_TIMEOUT=6h
        ;;
    train_full)
        # The declared aligned-11768 arm: the H008 anchor budget exactly,
        # 11768 x 1,048,576 = 12,339,642,368 tokens. CKPT_STEPS=5884 gives the
        # mid-run (un-annealed) point H008 also has, so both comparisons are
        # schedule-matched. The H008 anchor ran 11768 steps in 6.02 h on 2x8;
        # 9h of timeout leaves room for a slow start without reaching the 8 h
        # minRuntime shield's preemption window unnecessarily early.
        # TRAIN_FULL_STEPS (H038) moves the step count with the mixture's token count; the
        # mid-run checkpoint stays at the half-way step.
        # EPHEMERAL_STEPS must stay below CKPT_STEPS (= STEPS/2), or OLMo-core refuses the config
        # ("ephemeral_save_interval must be less than save_interval"): H054's 176- and 490-step
        # continued-SFT arms pass EPHEMERAL_STEPS=-1 (off). The 1024 default is unchanged for long runs.
        export STEPS="${TRAIN_FULL_STEPS:-11768}" NNODES=2 NPROC=8 EPHEMERAL_STEPS="${EPHEMERAL_STEPS:-1024}"
        export CKPT_STEPS=$(( STEPS / 2 ))
        export JOB_TIMEOUT="${JOB_TIMEOUT:-9h}"
        # RUN_TAG keeps the run name and output dir distinct from the 3072-update
        # arm, whose dir already exists; MODE itself must read "train" downstream.
        # Two permanent checkpoints, not one: KEEP_LAST_N=1 would prune step5884
        # the moment step11768 lands, and step5884 is the schedule-matched
        # mid-run comparison against the H008 anchor. 2 x 207 GB; convert and
        # delete each as it lands.
        export KEEP_LAST_N=2
        RUN_TAG="train-full-s${DATA_LOADER_SEED}"
        BUDGET=full
        MODE=train
        ;;
    tokenize)
        # CPU-only; builds the arm's own numpy cache (THINK_TOKENS changes its key). Not
        # ceres: this caller is non-preemptible (minRuntime set), and ceres rejects that from
        # ai2/olmo-instruct, which has no allocation there.
        export TOKENIZE_CLUSTERS="${TOKENIZE_CLUSTERS:-ai2/saturn ai2/neptune ai2/jupiter}"
        MODE=tokenize_full
        ;;
    gate)
        export STEPS=30 NNODES=1 NPROC=8 CKPT_STEPS=1000000 EPHEMERAL_STEPS=-1
        export JOB_TIMEOUT=45m
        ;;
    convert)
        # Both arms use the current converter; only legacy training uses H008's image.
        IMAGE="$BUILT_IMAGE"
        export JOB_TIMEOUT=2h CONVERT_GPUS=1 CONVERT_DEVICE=cuda CONVERT_CLUSTER=ai2/holmes
        export CONVERT_PYTHONPATH=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-anchor/olmo-core-b1fd2c97/src
        if [[ "$EXPERIMENT" != "h010" ]]; then
            # BUDGET=full converts the train_full dir of the same arm and seed.
            export CKPT_ROOT="${CKPT_ROOT:-$H015_ROOT/${EXPERIMENT}-${ARM}${BOUNDARY_TAG}-train${BUDGET:+-$BUDGET}-s${DATA_LOADER_SEED}}"
        else
            export CKPT_ROOT="${CKPT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-hillclimb-1895/${ARM}-train-20260918}"
        fi
        export STEP="${STEP:-step3072}"
        if [[ "$ARM" == "legacy" ]]; then
            CACHE=15bfc110a1-6068a350
        elif [[ "$ARM" == "think" ]]; then
            # The tokenizer saved in the think cache carries the promoted slots, and the
            # export must ship that one.
            CACHE="${THINK_CACHE:?set THINK_CACHE to the think arm numpy cache dir name}"
        elif [[ "$ARM" == olmo35* ]]; then
            # The olmo35 cache's tokenizer dir carries the Olmo 3.5 template the arm trained on.
            CACHE="${H038_CACHE:?set H038_CACHE to the olmo35 arm numpy cache dir name}"
        else
            CACHE=062b8a3d20-6068a350
        fi
        export CONVERT_TOKENIZER="/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache/numpy_sft/$CACHE/tokenizer"
        ;;
    *) echo "Expected train, train_full, tokenize, gate or convert" >&2; exit 1 ;;
esac
# LR is the anchor recipe's 5e-5 unless the caller sets it (H054's continued-SFT arms use 2e-5).
export BASE=hero-small-nonemo SEQ=65536
export LR="${LR:-5e-5}"
export THINK_TOKENS
# Train and gate must name their cache. Tokenize names it when it is already known, which
# makes a flag-off tokenize job a CPU preflight: the production hash either resolves to the
# arm's cache (and exits "nothing to do") or raises -- before any GPU is reserved.
if [[ -n "$ARM_CACHE" ]]; then
    export EXPECTED_NUMPY_CACHE="$ARM_CACHE"
elif [[ "$MODE" == "train" || "$MODE" == "gate" ]]; then
    echo "set THINK_CACHE (think) or H038_CACHE (olmo35 arms) to the arm's numpy cache dir name" >&2; exit 1
else
    unset EXPECTED_NUMPY_CACHE
fi
export DATA_LOADER_SEED
export CLUSTER=ai2/holmes WORKSPACE=ai2/olmo-instruct PREEMPTIBLE=0
export PRIORITY="${PRIORITY:-normal}"
# A 2x8 job needs EVERY replica ready within 10 min or Beaker cancels the group
# ("timed out after waiting 10m0s for synchronized replica start") -- which killed
# 01M332MXC7PKM3WWCHCW5WW4QM at 23:07 UTC on 2026-09-21 before step 1, replica 1
# having scheduled but never readied. Retries are the fix for that, but a retry
# RESTARTS the command, and olmo-core only resumes when RESUME_FROM is set -- so a
# mid-run retry silently trains from step 0 again. Default stays 0 for that reason;
# override to 1 when the run has not started yet and a start-time failure is the
# risk being managed, then watch the step counter on the retry.
export MAX_RETRIES="${MAX_RETRIES:-0}"
export KEEP_LAST_N="${KEEP_LAST_N:-1}"
if [[ "$EXPERIMENT" != "h010" ]]; then
    H015_TAG="${ARM}${BOUNDARY_TAG}-${MODE}${BUDGET:+-$BUDGET}-s${DATA_LOADER_SEED}"
    export RUN_NAME="${RUN_NAME:-hero-sft-${EXPERIMENT}-$H015_TAG}"
    export OUTPUT_DIR="${OUTPUT_DIR:-$H015_ROOT/${EXPERIMENT}-$H015_TAG}"
fi
export RUN_NAME="${RUN_NAME:-hero-sft-h010-${ARM}${BOUNDARY_TAG}-${RUN_TAG:-${MODE}-s${DATA_LOADER_SEED}}-20260918}"
export OUTPUT_DIR="${OUTPUT_DIR:-/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-hillclimb-1895/${ARM}${BOUNDARY_TAG}-${RUN_TAG:-${MODE}}-20260918}"
# Caller accounts for all queued/running jobs against 32 urgent + 32 normal.
# Explicit interpreter avoids local sync of the separately pinned MoE runtime.
export PY="${PY:-python}"
bash scripts/train/debug/oc_sft_olmoe3_kda_think.sh "$IMAGE" "$MODE"
