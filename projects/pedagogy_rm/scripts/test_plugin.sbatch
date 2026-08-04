#!/bin/bash
#SBATCH -J pedagogy_plugin
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=0:40:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_plugin_%j.out
#SBATCH -e logs/pedagogy_plugin_%j.err
#
# The offline probe and the online reward build their chat context in different
# files. They can drift apart - a different template call, a token boundary an
# index off - and nothing raises. The reward would keep returning numbers, they
# would simply stop being the numbers that were measured. This scores real
# labelled turns through the plugin and fails if they no longer track.
set -euo pipefail

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
# venv_oi, not the venv_test the labels and the probe were produced under: grpo_fast
# needs vllm's weight_transfer API and Python 3.11+, neither of which venv_test has.
# That moves transformers from 4.57 to 5.x underneath the head, which is exactly the
# drift this test exists to catch, so run it here before trusting the reward.
VENV=${VENV:-$SCRATCH/venv_oi}

export HF_HOME=$SCRATCH/hf-cache
export HF_HUB_CACHE=$SCRATCH/hf-cache/hub
export XDG_CACHE_HOME=$SCRATCH/xdg-cache
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor-cache
export HF_HUB_OFFLINE=1
export PEDAGOGY_GPU=1

source "$VENV/bin/activate"
cd "$REPO"
mkdir -p logs

python -u -m pytest projects/pedagogy_rm/test_plugin.py -x -q -p no:cacheprovider

echo "done $(date -Is)"
