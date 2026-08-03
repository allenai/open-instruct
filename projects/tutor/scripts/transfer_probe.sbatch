#!/bin/bash
#SBATCH -J transfer_probe
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
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
mkdir -p logs

REPO=${REPO:-$HOME/open-instruct}
DATA=${DATA:-$HOME/grpo_tutor}
STUDENT=${STUDENT:-Qwen/Qwen2.5-0.5B-Instruct}
PORT=${PORT:-8001}
# 0.82 keeps pairs like "simple interest" vs "simple and compound interest" and
# "rounding to tens" vs "rounding to hundreds", while excluding mere neighbours.
# Below ~0.75 the pairs stop being the same lesson; see build_pairs_by_similarity.
THRESHOLD=${THRESHOLD:-0.82}

cd "$REPO"

# The answering channel stays on a local vLLM even though the gateway has far
# better models: it scores options by length-normalised log-probability using
# echo+logprobs, which a chat gateway does not expose. Keeping this model pinned
# is also what makes the numbers comparable with the run history.
vllm serve "$STUDENT" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --disable-log-requests &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

until curl -sf "http://localhost:$PORT/v1/models" >/dev/null; do
    sleep 5
    kill -0 $SERVER 2>/dev/null || { echo "vllm died before becoming ready"; exit 1; }
done
echo "student up on :$PORT"

python -m projects.tutor.transfer_probe \
    --items data/items_all.jsonl \
    --units data/units_all.jsonl \
    --embeddings data/units_all.npz \
    --traces "$DATA/runs/gen_traces.jsonl" \
    --pair-threshold "$THRESHOLD" \
    --student-model "$STUDENT" \
    --student-url "http://localhost:$PORT/v1" \
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
