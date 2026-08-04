#!/bin/bash
#SBATCH -J pedagogy_smoke
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH -o logs/pedagogy_smoke_%j.out
#SBATCH -e logs/pedagogy_smoke_%j.err
#
# The real run behind a 1B policy and a handful of steps, to find the wiring bugs
# in minutes rather than in a twelve-hour allocation. Everything it exercises is
# shared with the real run: the RLVR files, the prompt format, the LoRA wrapping,
# merge_adapter into a CUDA IPC weight sync, the reference policy taken from the
# adapter-disabled base, the plugin loading inside the vLLM actor, the group
# scorer replacing the verifier score, and the eval pass.
#
# THE ENCODER STAYS 7B. It is named in data/head.npz rather than on the command
# line, so the reward here is computed by exactly the model the head was fitted
# on. That is the point - this checks the scorer, so the scorer must be real.
# What it deliberately does NOT check is whether the 7B policy fits alongside it.
set -euo pipefail

export POLICY=allenai/OLMo-2-0425-1B-Instruct
export LEARNERS=1
export ENGINES=1
export VLLM_UTIL=0.20
export PROMPTS=8
export EPISODES=192   # 8 prompts x 8 samples = 64 an episode, so 3 steps
export EVAL_EVERY=2
export SAVE_FREQ=1000
export EXP=pedagogy_smoke

# Not $(dirname $0): sbatch runs a spooled copy of this file, so $0 is somewhere
# under /var/spool and says nothing about where the repo is.
SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
export SCRATCH REPO
exec bash "$REPO/projects/pedagogy_rm/scripts/train.sbatch"
