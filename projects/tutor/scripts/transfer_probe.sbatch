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
# Do the two cheap steps FIRST, on a laptop, before queueing this:
#
#   1. annotate units (API only, no GPU):
#        python -m projects.tutor.units \
#            --items data/state_tests/train_items.jsonl \
#            --traces runs/gen_traces.jsonl \
#            --out data/units.jsonl
#      Read the coverage report. `pairable_items` is the number that matters;
#      below ~40 the probe cannot resolve the effect and this job is wasted.
#
#   2. eyeball the pairing (no GPU, no API):
#        python -m projects.tutor.transfer_probe ... --dry-run --show-pairs 20
#      If the pairs do not look like they test the same skill, the unit tags are
#      wrong and every number below inherits that.

set -euo pipefail
mkdir -p logs

REPO=${REPO:-$HOME/open-instruct}
DATA=${DATA:-$HOME/grpo_tutor}
STUDENT=${STUDENT:-Qwen/Qwen2.5-0.5B-Instruct}
PORT=${PORT:-8001}

cd "$REPO"

# The same answering student the anchor has always used. Keep it pinned: this
# channel is the instrument, and swapping it changes what every historical
# number means.
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
    --items "$DATA/data/state_tests/train_items.jsonl" \
    --units "$DATA/data/units.jsonl" \
    --traces "$DATA/runs/gen_traces.jsonl" \
    --student-model "$STUDENT" \
    --student-url "http://localhost:$PORT/v1" \
    --report \
    --show-pairs 10 \
    --out "$DATA/runs/transfer_probe.json"

# Reading the output:
#
#   specificity inside +/-2 SE  -> no transferable teaching signal. The outcome
#                                  measure has one channel and it is leakage.
#                                  Do not run another GRPO job against it; either
#                                  change the outcome measure or write up the
#                                  negative result.
#   specificity beyond 2 SE     -> a pedagogy channel exists that teacher_acc
#                                  cannot see. Rebuild the reward around transfer.
#
# Check the leaked/clean split FIRST. If leaked source dialogues move B as much
# as clean ones, the pairs share more than a knowledge unit and the headline
# means nothing.
