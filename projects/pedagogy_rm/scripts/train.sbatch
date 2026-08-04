#!/bin/bash
#SBATCH -J pedagogy_grpo
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_grpo_%j.out
#SBATCH -e logs/pedagogy_grpo_%j.err
#
# The MIT-cluster wrapper around scripts/train.sh: a Slurm header, a CUDA module, a
# venv, and the cache paths that have to live on scratch rather than in a home
# directory with a quota. The run itself is in train.sh, because the eduLLM platform
# execs a command string in a container and cannot use any of this.
#
# ONE H100 AND THEREFORE LORA. mit_preemptable caps a user at four GPUs, and full-weight
# 7B does not fit on one card whatever the stage: see the arithmetic in train.sh.
# train_full_4gpu.sbatch is the four-card full-weight variant on the same cluster.
set -euo pipefail

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}

export SCRATCH CACHE_ROOT=$SCRATCH
# The weights are already on scratch and the compute nodes have no route to the hub.
export HF_HUB_OFFLINE=1

export MODE=${MODE:-lora}
export GPUS=${GPUS:-1}
export LEARNERS=${LEARNERS:-1}
export ENGINES=${ENGINES:-1}

# DeepSpeed will not import at all without a CUDA toolkit: its op builder shells out to
# $CUDA_HOME/bin/nvcc while deciding which ops are compatible, and raises
# MissingCUDAException when the variable is unset. The venv cannot supply one - torch
# ships CUDA runtime libraries but no compiler. cuda/13.0.1 matches torch 2.11.0+cu130
# in both major and minor; a 12.x module would import and then warn on every JIT op.
# set +u because lmod's own shell functions read unset variables.
set +u
command -v module >/dev/null 2>&1 || source /usr/share/lmod/lmod/init/bash
module load cuda/13.0.1
source "$VENV/bin/activate"
set -u

cd "$REPO"
mkdir -p logs output
export PYTHONPATH=$REPO

bash projects/pedagogy_rm/scripts/train.sh

echo "done $(date -Is)"
