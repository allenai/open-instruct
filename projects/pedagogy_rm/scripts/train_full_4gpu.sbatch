#!/bin/bash
#SBATCH -J pedagogy_grpo_full
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --time=12:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=32
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_grpo_full_%j.out
#SBATCH -e logs/pedagogy_grpo_full_%j.err
#
# Full-weight GRPO on OLMo-2-7B-Instruct, sharded across four H100s. Four is the
# ceiling: mit_preemptable's QOS caps gres/gpu at 4 per user, so this is the largest
# full-weight run this cluster allows.
#
# THREE LEARNERS AND ONE vLLM ENGINE. The split is forced rather than chosen - vLLM
# needs a card of its own for generation, and whatever is left trains. Roughly where an
# 80GB learner card goes under ZeRO-3 with three learners:
#
#   sharded bf16 weights          14.6 / 3  =  4.9GB
#   sharded bf16 gradients        14.6 / 3  =  4.9GB
#   sharded fp32 Adam + master    87.6 / 3  = 29.2GB
#   sharded reference policy      14.6 / 3  =  4.9GB
#   checkpointed activations, all-gather buffers, fragmentation  ~8GB
#                                              -------
#                                              ~52GB
#
# So it fits on an 80GB card with room to raise the micro-batch, and OFFLOAD stays 0.
# On a 48GB card the Adam row alone would leave nothing, which is why the platform
# variant of this run sets OFFLOAD=1 on 4xL40S. The vLLM card holds a 14.6GB copy of
# the policy plus KV cache at 0.55 of 80GB, and the scorer's truncated frozen encoder
# at about 9GB in the same process.
#
# THE REWARD IS STILL COMPUTED BY A FROZEN ENCODER, which is what keeps full-weight
# training honest here rather than merely affordable. train.sh carries the argument.
set -euo pipefail

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

export MODE=full
export GPUS=4
export LEARNERS=3
export ENGINES=1
export TP=1
export OFFLOAD=0
export EXP=${EXP:-pedagogy_olmo7b_full}
# Three learners at one sequence each is 3 sequences a backward pass. The rollout is
# unchanged from the LoRA run so the two are comparable.
export MICRO_BATCH=${MICRO_BATCH:-1}

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
export SCRATCH REPO
exec bash "$REPO/projects/pedagogy_rm/scripts/train.sbatch"
