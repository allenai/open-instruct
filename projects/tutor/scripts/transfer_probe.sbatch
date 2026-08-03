#!/bin/bash
#SBATCH -J transfer_probe
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/transfer_probe_%j.out
#SBATCH -e logs/transfer_probe_%j.err

# Does tutoring on problem A move the student on a different problem B?
#
# Inference only - no tutor generation, no training - so one preemptable H100 is
# enough and a requeue costs minutes rather than a run.
#
# PREPARE ON A LAPTOP FIRST. Both steps need the TrueFoundry gateway (TFY_API_KEY
# or TFY_API in .env) and no GPU at all:
#
#   1. annotate + embed both item pools, ~2 min each:
#        python -m projects.tutor.units \
#            --items $DATA/data/state_tests/train_items.jsonl \
#            --traces $DATA/runs/gen_traces.jsonl --out data/units.jsonl
#        python -m projects.tutor.units \
#            --items $DATA/data/state_tests/eval_items.jsonl \
#            --out data/units_eval.jsonl --no-canonicalize
#      then merge to data/units_all.jsonl and embed once to data/units_all.npz.
#
#      Both pools, because only the SOURCE needs a dialogue - the target just has
#      to be an item. The 235 held-out eval items therefore roughly double the
#      pairing pool for free, and using held-out items as targets is a cleaner
#      generalisation test than reusing the training set.
#
#   2. eyeball the pairing, no GPU and no API:
#        python -m projects.tutor.transfer_probe ... --dry-run --show-pairs 20
#      Wrong unit tags are silent and everything downstream inherits them.

set -euo pipefail

SCRATCH=${SCRATCH:-$HOME/orcd/scratch}
REPO=${REPO:-$SCRATCH/open-instruct}
DATA=${DATA:-$SCRATCH/grpo_tutor}
VENV=${VENV:-$SCRATCH/venv_v4}
# Pinned. run_v4.md: "student (answers) Qwen2.5-0.5B-Instruct, frozen - pinned so
# teacher_acc stays comparable with v0-v3". Changing it changes what every
# historical number means.
export STUDENT=${STUDENT:-Qwen/Qwen2.5-0.5B-Instruct}

# Everything cache-shaped goes to scratch. HOME is a quota that other projects
# keep full, and a job that overruns it dies during vLLM engine init with
# "Disk quota exceeded", which reads like a vLLM bug and is not one. Same set of
# variables as grpo_tutor/scripts/env.sh, which exists because of that failure.
export HF_HOME=$SCRATCH/hf-cache
export HF_HUB_CACHE=$SCRATCH/hf-cache/hub
export XDG_CACHE_HOME=$SCRATCH/xdg-cache
export XDG_CONFIG_HOME=$SCRATCH/xdg-config
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor-cache
# The student is already in the cache; compute nodes should not reach out.
export HF_HUB_OFFLINE=1

source "$VENV/bin/activate"
mkdir -p logs runs
# 0.82 keeps pairs like "simple interest" vs "simple and compound interest" and
# "rounding to tens" vs "rounding to hundreds", while excluding mere neighbours.
# Below ~0.75 the pairs stop being the same lesson; see build_pairs_by_similarity.
THRESHOLD=${THRESHOLD:-0.82}

cd "$REPO"

# No server. This job only scores options - it never generates - so serving the
# model adds a version-fragile dependency for nothing. Two attempts proved the
# point on this cluster: vLLM 0.25.1 rejects --disable-log-requests, and once
# that was fixed its kernel warmup imports MiniMax-M3 and Triton fails to parse
# the kernel source. A 0.5B transformers forward pass has neither problem, and
# it is the same arithmetic grpo_tutor/src/zpd_filter.py has always used.
python -u -m projects.tutor.transfer_probe \
    --items data/items_all.jsonl \
    --units data/units_all.jsonl \
    --embeddings data/units_all.npz \
    --traces "$DATA/runs/gen_traces.jsonl" \
    --pair-threshold "$THRESHOLD" \
    --backend local \
    --student-model "$STUDENT" \
    --report \
    --show-pairs 10 \
    --out "runs/transfer_probe_${SLURM_JOB_ID:-local}.json"

# READING THE OUTPUT
#
# Read the contamination check FIRST. If leaked source dialogues move B as much
# as clean ones, the pairs share more than a knowledge unit and the headline
# means nothing - tighten --pair-threshold and rerun before interpreting.
#
# Then read P(gold), not the binary line. Both come from the same forward passes,
# but the binary indicator only moves when a hint flips the argmax, so it cannot
# resolve anything below about 0.11 at this n, while P(gold) resolves ~0.027.
# A plausible transfer effect is a couple of points, which is invisible to the
# first measure and visible to the second.
#
#   specificity within +/-2 SE  -> no transferable teaching. The outcome measure
#                                  has one channel and it is leakage. Change the
#                                  outcome or write up the negative result; do
#                                  not run another GRPO job against it.
#   specificity beyond 2 SE     -> a pedagogy channel exists that teacher_acc
#                                  cannot see. Rebuild the reward around transfer.
