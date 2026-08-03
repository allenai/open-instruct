#!/bin/bash
#SBATCH -J pedagogy_gen
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:2
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
# One card each. They do not co-exist on one; see the note above the launches.
# Serving rather than running in-process because
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

# ONE MODEL PER CARD, because two vLLM servers cannot share one. Its
# --gpu-memory-utilization is a fraction of TOTAL device memory, and the two
# checks it applies pull in opposite directions once another server is resident:
# startup demands util*total <= free, while cache sizing subtracts ALL current
# device usage (the other server's included) from util*total. With a 7B holding
# 38GB of an 80GB card, the second server needs util above 0.54 to have any cache
# and at most 0.52 to start. There is no value that satisfies both, and the two
# failures look unrelated - "No available memory for the cache blocks" one way,
# "Free memory on device is less than desired utilization" the other. Do not
# spend another job trying to thread it.
CUDA_VISIBLE_DEVICES=0 vllm serve "$TEACHER" --port 8000 --gpu-memory-utilization 0.85 \
    --max-model-len 4096 --disable-log-stats &
TEACHER_PID=$!
CUDA_VISIBLE_DEVICES=1 vllm serve "$STUDENT" --port 8001 --gpu-memory-utilization 0.40 \
    --max-model-len 4096 --disable-log-stats &
STUDENT_PID=$!

wait_for() {  # port, pid, name — watch THAT server's pid, so a dead one fails fast
    local port=$1 pid=$2 name=$3
    for _ in $(seq 1 120); do
        curl -sf "http://localhost:$port/v1/models" >/dev/null 2>&1 && { echo "$name up on :$port"; return 0; }
        kill -0 "$pid" 2>/dev/null || { echo "$name died before serving; see the .err log"; return 1; }
        sleep 10
    done
    echo "$name never came up on :$port"
    return 1
}
wait_for 8000 "$TEACHER_PID" teacher || exit 1
wait_for 8001 "$STUDENT_PID" student || exit 1

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
