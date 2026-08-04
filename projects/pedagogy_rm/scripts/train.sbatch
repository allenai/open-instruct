#!/bin/bash
#SBATCH -J pedagogy_grpo
# mit_preemptable DESPITE THE PREEMPTIONS, BECAUSE THE ALTERNATIVE NEVER STARTS.
# mit_normal_gpu is the only non-preemptable H100 partition here and it is a single node,
# so a job there waits behind everyone else indefinitely; mit_preemptable has twenty-seven.
# A run that is preempted and resumes beats a run that is still queued.
#
# WHAT THAT COSTS AND HOW IT IS PAID. Two runs were taken inside fifteen minutes, at 2:13
# and again around step 3, and Slurm requeues this script from the top - so the cost of a
# preemption is every step since the last checkpoint, plus the three minutes it takes to
# load a 7B into both the trainer and vLLM. That makes the checkpoint interval the number
# that decides whether this run progresses at all, which is why the real run passes a small
# SAVE_FREQ rather than leaving the default: at fifty steps it would be preempted long
# before writing anything and would restart from zero forever.
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
# Kept even though these nodes can reach the hub, because every checkpoint this run needs
# is already under $SCRATCH/hf-cache and pinning it offline makes the run independent of
# hub availability rather than merely faster. The cost is that a cache miss fails instead
# of downloading, which is the right way round for a run that has to be reproducible: a
# missing weight should stop it, not silently fetch a different revision.
export HF_HUB_OFFLINE=1

export MODE=${MODE:-lora}
export GPUS=${GPUS:-1}
export LEARNERS=${LEARNERS:-1}
export ENGINES=${ENGINES:-1}

# W&B ONLINE, BECAUSE THESE COMPUTE NODES HAVE A ROUTE OUT. Measured rather than assumed,
# on node2500 of this partition: pypi.org returns 200, api.wandb.ai and huggingface.co both
# answer, and no proxy variables are set. An earlier version of this file said the opposite
# and wrote WANDB_MODE=offline on that belief; what had actually happened was that
# HF_HUB_OFFLINE below made huggingface_hub refuse its own request, which reads exactly
# like a missing network.
#
# Online matters here beyond convenience: a run of this length is worth watching while it
# runs, and offline mode produces a directory nobody looks at until it is over. Credentials
# come from ~/.netrc, which is on the shared home directory and so is already on the node.
# Set WANDB_MODE=offline to get the old behaviour back if the network does go away.
#
# eduLLM RATHER THAN THE PERSONAL ENTITY, so these land where the group can see them and
# beside anything the AWS platform runs - it injects WANDB_ENTITY=eduLLM for the same reason.
# Checked before setting rather than assumed: `wandb.Api().viewer.teams` lists
# zsophia-massachusetts-institute-of-technology, aimosprite and eduLLM, so this account can
# write here. That check matters because wandb.init against a team you are not on fails at
# startup, after the queue wait and the model load.
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=${WANDB_DIR:-$SCRATCH/wandb}
export WANDB_PROJECT=${WANDB_PROJECT:-pedagogy-rm}
export WANDB_ENTITY=${WANDB_ENTITY:-eduLLM}
mkdir -p "$WANDB_DIR"

# PROVENANCE, SO A CURVE CAN BE TRACED BACK TO CODE AND DATA RATHER THAN TO AN AFTERNOON.
# A W&B run records its arguments and nothing about what produced them, and this tree is
# edited between runs, so without this a result six weeks old is unattributable: the reward
# depends on which head.npz was loaded and on plugin.py's aggregation as much as on any flag.
#
# The dirty count is the honest half. Files edited and not committed mean the commit named
# here does not describe what ran, and saying so on the run is worth more than a tidy tag
# that is wrong - which is the state every run before this one was in.
COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
DIRTY=$(git -C "$REPO" status --porcelain 2>/dev/null | grep -cvE '^\?\? (logs|output|data)/' || true)
HEAD_SHA=$(sha256sum "$REPO/data/head.npz" 2>/dev/null | cut -c1-12)
DATA_SHA=$(sha256sum "$REPO/data/rl/train.jsonl" 2>/dev/null | cut -c1-12)
export WANDB_TAGS="commit-$COMMIT,head-$HEAD_SHA,${MODE},${SCORER:-pedagogy}"
export WANDB_NOTES="code $COMMIT (uncommitted files: $DIRTY) | head.npz $HEAD_SHA | train.jsonl $DATA_SHA | $(hostname)"
echo "provenance: $WANDB_NOTES"
if [ "$DIRTY" != "0" ]; then
    echo "WARNING: $DIRTY files differ from $COMMIT, so this run is NOT reproducible from it" >&2
fi

# Trainer state, so the requeue this partition's preemptions cause resumes rather than
# restarting. train.sh appends the experiment name and carries the reasoning. A smoke run
# overrides this to empty, because three steps are cheaper to repeat than to checkpoint.
export CKPT_ROOT=${CKPT_ROOT:-$SCRATCH/ckpt}

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

# TRITON COMPILES A C EXTENSION THE FIRST TIME ANY KERNEL LAUNCHES, and this cluster's
# python3.12 ships without headers: there is no python3-devel and /usr/include/python3.12
# does not exist, so the build fails with `Python.h: No such file or directory` from
# inside a vLLM engine, which surfaces as `Engine core initialization failed` several
# hundred lines away from the cause.
#
# The headers come from a uv-managed CPython instead. It is 3.12.13, the same patch
# release as the venv's interpreter, so these are the right headers rather than merely
# compatible ones. CPATH rather than a flag, because the compiler invocation belongs to
# Triton: it passes -I from sysconfig, which points at the directory that does not exist,
# and gcc searches CPATH after that without needing to be told.
PY_INCLUDE=${PY_INCLUDE:-$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12}
if [ -f "$PY_INCLUDE/Python.h" ]; then
    export CPATH="$PY_INCLUDE${CPATH:+:$CPATH}"
else
    echo "warning: no Python.h at $PY_INCLUDE; Triton will fail to build its C extension" >&2
fi

# vLLM 0.21 encodes engine output with msgspec and refuses torch.dtype, which its own
# KV-cache specs contain: the engine starts, reports the API server ready, and then the
# thread returning results dies with `Object of type <class 'torch.dtype'> is not
# serializable`. Nothing of ours puts a dtype on that wire - open_instruct's weight
# transfer already sends dtype *names* - and vllm_utils.py carries a monkey-patch for the
# same family of bug in MambaSpec, so this is that bug in the output path.
#
# The flag is vLLM's own documented fallback and it names the risk honestly: it allows
# pickle on that socket. The socket is 127.0.0.1 inside one Slurm allocation, so the
# thing being trusted is this job's own engine process.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd "$REPO"
mkdir -p logs output
export PYTHONPATH=$REPO

bash projects/pedagogy_rm/scripts/train.sh

echo "done $(date -Is)"
