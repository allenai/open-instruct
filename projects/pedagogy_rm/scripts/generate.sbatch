#!/bin/bash
#SBATCH -J pedagogy_gen
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_gen_%j.out
#SBATCH -e logs/pedagogy_gen_%j.err
#
# Generate tutor/student dialogues for labelling. No reward, no judge, no
# training - this produces text for humans to rate, and nothing else.
#
# Both models are served from one card: OLMo-2-7B and a 1.5B student fit
# together in 80GB at these utilisations. Serving rather than in-process because
# this job DOES generate (unlike the probes, which only scored), and generation
# through vLLM's server is what the async client in generate.py expects.
set -euo pipefail

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

SCRATCH=${SCRATCH:-$HOME/orcd/scratch}
REPO=${REPO:-$SCRATCH/open-instruct}
DATA=${DATA:-$SCRATCH/grpo_tutor}
# venv_test (vLLM 0.11.2), NOT venv_v4 (0.25.1). 0.25.1 cannot serve anything on
# this cluster: its kernel warmup imports MiniMax-M3 and Triton fails to parse
# the kernel source at import time, so the engine never starts. Three jobs died
# that way before this comment existed. venv_test is what every working
# grpo_tutor run used.
VENV=${VENV:-$HOME/venv_test}
TEACHER=${TEACHER:-allenai/OLMo-2-1124-7B-Instruct}
STUDENT=${STUDENT:-Qwen/Qwen2.5-1.5B-Instruct}
ITEMS=${ITEMS:-$DATA/data/state_tests/train_items.jsonl}
N_ITEMS=${N_ITEMS:-300}
K=${K:-2}

# HOME is a 200GB quota; overrunning it kills the job at model load with
# "Disk quota exceeded". Everything cache-shaped goes to scratch.
export HF_HOME=$SCRATCH/hf-cache
export HF_HUB_CACHE=$SCRATCH/hf-cache/hub
export XDG_CACHE_HOME=$SCRATCH/xdg-cache
export XDG_CONFIG_HOME=$SCRATCH/xdg-config
export VLLM_CACHE_ROOT=$SCRATCH/vllm-cache
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor-cache

source "$VENV/bin/activate"
cd "$REPO"
mkdir -p logs runs

# Pre-download on the login node instead if the compute nodes have no route out:
#   hf download allenai/OLMo-2-1124-7B-Instruct
# then set HF_HUB_OFFLINE=1 here.

cleanup() { kill "${TEACHER_PID:-}" "${STUDENT_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

# --gpu-memory-utilization is a fraction of TOTAL device memory, not of what is
# still free, and vLLM counts memory the other server already took against it.
# So the SECOND server needs a budget covering the first's share as well: 0.45
# then 0.85 leaves the student ~32GB of an 80GB card, not 0.85 of it. Setting
# the intuitive 0.60/0.25 fails with "No available memory for the cache blocks",
# which reads like the student is too big and is nothing of the sort.
vllm serve "$TEACHER" --port 8000 --gpu-memory-utilization 0.45 \
    --max-model-len 4096 --disable-log-stats &
TEACHER_PID=$!
# Wait for the teacher to finish profiling before the student starts measuring,
# or both size their caches against a GPU that is about to change underneath them.
sleep 90
vllm serve "$STUDENT" --port 8001 --gpu-memory-utilization 0.85 \
    --max-model-len 4096 --disable-log-stats &
STUDENT_PID=$!

for port in 8000 8001; do
    for _ in $(seq 1 120); do
        curl -sf "http://localhost:$port/v1/models" >/dev/null 2>&1 && break
        kill -0 $TEACHER_PID 2>/dev/null || { echo "teacher died before ready"; exit 1; }
        sleep 10
    done
    curl -sf "http://localhost:$port/v1/models" >/dev/null || { echo "port $port never came up"; exit 1; }
    echo "up on :$port"
done

# One client, two ports: generate.py routes by model name, so the student is
# reached through the teacher's base URL only if both are served together. They
# are not, so the student goes through its own port via --student-base-url.
python -u -m projects.pedagogy_rm.generate \
    --items "$ITEMS" \
    --out "runs/traces_${SLURM_JOB_ID:-local}.jsonl" \
    --teacher-model "$TEACHER" \
    --student-model "$STUDENT" \
    --base-url http://localhost:8000/v1 \
    --student-base-url http://localhost:8001/v1 \
    --n-items "$N_ITEMS" --k "$K" --turns 3

echo "done $(date -Is)"

# NEXT, and not before: build slices with overlap, label them, and measure
# per-dimension agreement. A dimension whose raters disagree cannot be learned
# by any probe, so that number decides what the rest of the project is for.
#
#   python -m projects.pedagogy_rm.build_label_set \
#       --traces runs/traces_*.jsonl --out-dir data/label_slices --n 600 --raters 4
