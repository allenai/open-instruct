#!/bin/bash
#SBATCH -J hint_ladder
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=1:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/hint_ladder_%j.out
#SBATCH -e logs/hint_ladder_%j.err
#
# Can anything short of the answer move the student on its own problem?
#
# Runs the ladder on both students. The 0.5B is the one every run in the history
# used, so it is the comparable number. The 3B is the test of whether that
# student was simply too weak to hold the partial understanding a hint completes
# - it cannot be read from the 0.5B alone, because "no effect" and "no capacity
# for an effect" look identical on one model.
#
# Scoring only, no generation and no training, so there is no server: see
# transfer_probe.sbatch for why that matters on this cluster.
set -euo pipefail

SCRATCH=${SCRATCH:-$HOME/orcd/scratch}
REPO=${REPO:-$SCRATCH/open-instruct}
DATA=${DATA:-$SCRATCH/grpo_tutor}
VENV=${VENV:-$SCRATCH/venv_v4}
MODELS=${MODELS:-"Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-3B-Instruct"}

# HOME is a quota other projects keep full, and overrunning it kills the job at
# model load with "Disk quota exceeded". Everything cache-shaped goes to scratch.
export HF_HOME=$SCRATCH/hf-cache
export HF_HUB_CACHE=$SCRATCH/hf-cache/hub
export XDG_CACHE_HOME=$SCRATCH/xdg-cache
export XDG_CONFIG_HOME=$SCRATCH/xdg-config
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor-cache
# Both students are already cached; compute nodes should not reach out.
export HF_HUB_OFFLINE=1

source "$VENV/bin/activate"
cd "$REPO"
mkdir -p logs runs

for model in $MODELS; do
    echo
    echo "############ $model ############"
    python -u -m projects.tutor.hint_ladder \
        --items data/items_all.jsonl \
        --units data/units_all.jsonl \
        --traces "$DATA/runs/gen_traces.jsonl" \
        --student-model "$model" \
        --out "runs/hint_ladder_$(basename "$model")_${SLURM_JOB_ID:-local}.json"
done

# READING THE OUTPUT
#
# Check "answer stated" FIRST, on each model separately. It is the ceiling, not
# a result. If it is small, that student cannot register a hint of any size and
# its other four rungs are noise - say nothing about teaching from them.
#
# Then read "real dialogue", which is the training objective written as a number.
# If it is flat on the 0.5B but live on the 3B, the student was the bottleneck
# and the setup is worth keeping. If it is flat on both while "answer stated" is
# large on both, then telling is the only thing this environment can register,
# any reward built on student correctness is maximised by leaking, and the
# conclusion is to change the environment rather than the reward model.
#
# Rungs 1 and 2 are the tiebreak. If a bare sentence moves the student where a
# whole dialogue does not, the channel is open and the teacher's output is the
# weak part - which is a much easier problem than the setup being wrong.
