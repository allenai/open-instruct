#!/bin/bash
#SBATCH -J pedagogy_hidden
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_hidden_%j.out
#SBATCH -e logs/pedagogy_hidden_%j.err
#
# OLMo hidden states for every labelled turn. One forward pass each, no
# generation, so this runs in-process through transformers rather than through
# vLLM - vLLM exists to make generation fast and would only add a server to fail.
set -euo pipefail

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$HOME/venv_test}
UNITS=${UNITS:-data/label_slices/slice_1.json}
OUT=${OUT:-data/hidden.npz}

export HF_HOME=$SCRATCH/hf-cache
export HF_HUB_CACHE=$SCRATCH/hf-cache/hub
export XDG_CACHE_HOME=$SCRATCH/xdg-cache
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor-cache
export HF_HUB_OFFLINE=1

source "$VENV/bin/activate"
cd "$REPO"
mkdir -p logs data

python -u -m projects.pedagogy_rm.extract_hidden --units "$UNITS" --out "$OUT"

echo "done $(date -Is)"
