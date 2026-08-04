#!/bin/bash
#SBATCH -J pedagogy_grpo_z
# THE NON-PREEMPTABLE ARM, ON A CARD THAT IS SCARCER BUT CANNOT BE TAKEN AWAY.
# mit_normal_gpu does not preempt and caps at six hours; mit_preemptable gives two days and
# took two runs inside fifteen minutes. Neither is strictly better, so both are running: the
# preemptable arm has the time to finish 120 steps if it survives, and this one has the
# certainty to finish whatever fits in six hours.
#
# h200 SPECIFICALLY, AND THAT IS WHY THIS MAY QUEUE. At the time of writing two H200s were
# free across the partition against 86 free L40S, so this waits where a 48GB request would
# not. It is worth the wait: an H200 is 141GB, so the learner, vLLM and the scorer's frozen
# encoder share a card with room to spare, where the same three on an 80GB H100 come to
# about 55GB and on a 48GB L40S would not fit at all with vLLM's reservation.
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_grpo_z_%j.out
#SBATCH -e logs/pedagogy_grpo_z_%j.err
#
# WHAT IS DIFFERENT BESIDES THE MACHINE, WHICH IS THE POINT OF RUNNING IT. This arm scores
# with `pedagogy_z`, which z-scores each dimension inside the group before averaging, so the
# four contribute equally instead of in proportion to their spread. The raw mean leans about
# 1.5x harder on `elicits` than on `targeted`, and `targeted` is the dimension surface
# features cannot predict - so the two arms differ in whether the reward is dominated by the
# dimension that is easiest to game. plugin.py's `normalized` carries the full argument.
#
# Its reward curve is therefore NOT comparable to the other arm's: a z-scored reward is
# relative to the seven other samples of the same prompt and has no absolute scale. Compare
# the two through their `dim_*` metrics, which are the same rubric numbers in both, and
# through compare_policies.py on the sampled turns.
set -euo pipefail

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

export SCORER=pedagogy_z
export EXP=${EXP:-pedagogy_olmo7b_z}
export WANDB_PROJECT=${WANDB_PROJECT:-pedagogy-rm}
# Six hours, so fewer steps than the other arm rather than the same number more slowly. The
# checkpoint state means a second job continues from where the wall clock stopped this one.
export EPISODES=${EPISODES:-30720}
export SAVE_FREQ=${SAVE_FREQ:-10}

# SPENDING THE H200's EXTRA 61GB ON SPEED, WHICH IS THE ONLY WAY IT BUYS ANY. An H200 is
# the same GH100 die as an H100 - identical bf16 throughput - with 141GB against 80GB and
# 4.8TB/s against 3.35. So the card is about 1.4x on the memory-bound half, which is token
# generation, and exactly 1.0x on the compute-bound half, which is the forward and backward.
# Everything beyond that has to be bought with the memory.
#
# Gradient checkpointing off, because it exists to trade compute for memory and this card
# has memory: every checkpointed block is currently recomputed in the backward, and not
# doing that is worth roughly a fifth to a third of the backward.
#
# Micro-batch 4 rather than 1, so a backward covers four sequences instead of one.
#
# NEITHER CHANGES THE UPDATE, which is what keeps this an arm of the same experiment rather
# than a different one. --num_mini_batches is 1, so the optimizer step covers the whole
# rollout whatever the micro-batch, and lora_dropout is 0, so there is no dropout draw for
# the batch boundaries to move. What they do change is step time, so the two arms' wall
# clocks are no longer comparable - only their curves and their sampled turns are.
#
# Roughly where 141GB goes: learner base 14.6, adapters and their Adam under 1, activations
# without checkpointing at micro-batch 4 maybe 20, vLLM at 0.30 of the card 42, and the
# scorer's truncated encoder 9. Call it 87GB, which leaves real margin - VLLM_UTIL is a
# fraction, so vLLM's share grew from 24GB to 42GB by moving cards and nothing else.
export MICRO_BATCH=${MICRO_BATCH:-4}
export GRAD_CKPT=${GRAD_CKPT:-0}

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
export SCRATCH REPO
exec bash "$REPO/projects/pedagogy_rm/scripts/train.sbatch"
