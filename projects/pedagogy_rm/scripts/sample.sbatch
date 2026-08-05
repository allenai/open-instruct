#!/bin/bash
#SBATCH -J pedagogy_sample
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_sample_%j.out
#SBATCH -e logs/pedagogy_sample_%j.err
#
# Draw held-out turns from every policy under comparison and score them with the probe,
# so build_blind_set.py has something to blind and a human has something to rate.
#
# ONE TURN PER PROMPT PER ARM, not several. The comparison is paired by moment, and a
# moment sampled twice from one arm and once from another would weight it by how often it
# happened to be drawn. --samples 1 keeps each moment worth the same everywhere.
#
# TEMPERATURE 1.0, MATCHING TRAINING. A policy read greedily is not the policy the reward
# was computed on, and the question is about the thing that was trained.
#
# THE THREE ARMS ARE base, A@120 AND B@120. Five conditions was the other option - adding
# step 30 from each to locate an overoptimization peak - and it loses to arithmetic: about
# fifty human labels over five conditions is ten turns each, which resolves nothing. Three
# conditions at sixty moments is a real comparison, and the step-30 adapters are still on
# disk when there is appetite for a second pass.
set -euo pipefail

if [ -z "${PEDAGOGY_SNAPSHOT:-}" ]; then
    snapshot="${TMPDIR:-/tmp}/sample_${SLURM_JOB_ID:-$$}.sbatch"
    cp "${BASH_SOURCE[0]}" "$snapshot"
    export PEDAGOGY_SNAPSHOT="$snapshot"
    exec bash "$snapshot"
fi

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}
MOMENTS=${MOMENTS:-60}

ARM_A=${ARM_A:-$REPO/output/pedagogy_olmo7b/pedagogy_olmo7b__1__1785876035_checkpoints/step_120}
ARM_B=${ARM_B:-$REPO/output/pedagogy_olmo7b_z/pedagogy_olmo7b_z__1__1785876212_checkpoints/step_120}

export HF_HOME=$SCRATCH/hf-cache HF_HUB_CACHE=$SCRATCH/hf-cache/hub HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=$SCRATCH/xdg-cache TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor-cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

set +u
command -v module >/dev/null 2>&1 || source /usr/share/lmod/lmod/init/bash
module load cuda/13.0.1
source "$VENV/bin/activate"
set -u

PY_INCLUDE=$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12
[ -f "$PY_INCLUDE/Python.h" ] && export CPATH="$PY_INCLUDE${CPATH:+:$CPATH}"

# See train.sbatch for the full story. Short version: flashinfer caches JIT kernels under
# $HOME, the nodes do not all carry the same libstdc++, and a kernel built on one node
# fails to load on another with a GLIBCXX version error. The spack libstdc++ supplies the
# newer symbols and the per-node cache key stops the mismatch arising at all.
GCC_LIB=${GCC_LIB:-/orcd/software/core/001/spack/pkg/gcc/12.2.0/yt6vabm/lib64}
[ -d "$GCC_LIB" ] && export LD_LIBRARY_PATH="$GCC_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-$SCRATCH/flashinfer-cache/$(hostname -s)}
mkdir -p "$FLASHINFER_CACHE_DIR"

cd "$REPO"
export PYTHONPATH=$REPO
mkdir -p logs data/samples

# generate and score are separate processes on purpose: vLLM reserves a fraction of the
# card up front and the probe's encoder wants ~9GB of what is left. See sample_policy.py.
draw () {
    tag=$1; adapter=$2
    echo "=== $tag ==="
    python -u projects/pedagogy_rm/sample_policy.py generate \
        --prompts data/rl/eval.jsonl --limit "$MOMENTS" --samples 1 --temperature 1.0 \
        --policy allenai/OLMo-2-1124-7B-Instruct ${adapter:+--adapter "$adapter"} \
        --tag "$tag" --out "data/samples/${tag}.json"
    python -u projects/pedagogy_rm/sample_policy.py score --units "data/samples/${tag}.json"
}

draw base ""
draw arm_a "$ARM_A"
draw arm_b "$ARM_B"

python -u projects/pedagogy_rm/build_blind_set.py \
    --arm base=data/samples/base.json \
    --arm arm_a=data/samples/arm_a.json \
    --arm arm_b=data/samples/arm_b.json \
    --out-dir data/eval --moments 0

echo "done $(date -Is)"
