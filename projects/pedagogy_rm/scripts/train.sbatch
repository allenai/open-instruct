#!/bin/bash
#SBATCH -J pedagogy_grpo
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --time=12:00:00
#SBATCH --mem=800G
#SBATCH --cpus-per-task=64
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_grpo_%j.out
#SBATCH -e logs/pedagogy_grpo_%j.err
#
# GRPO on OLMo-2-7B-Instruct, rewarded by the hidden-state probe.
#
# FULL WEIGHT, because grpo_fast.py has no LoRA: use_peft is declared in
# model_utils.py and read nowhere in the training loop, and wiring PEFT through
# DeepSpeed and the vLLM weight sync is a change to the loop rather than a flag.
#
# THE ENCODER IS A SEPARATE FROZEN COPY, loaded by the scorer inside each vLLM
# actor. It has to be: the head is a linear map on one checkpoint's activation
# space, so pointing it at the model being trained would let the policy raise its
# reward by moving its own representations without changing a word it writes.
# vllm_gpu_memory_utilization is held down to leave room for that copy - the
# scorer keeps only the blocks up to the deepest layer any head reads, so it is
# roughly 9GB rather than 14GB, but vLLM will still take everything it is offered.
#
# NO --chat_template_name AND NO --add_bos, both deliberate. Leaving the template
# unset makes open-instruct use the tokenizer's own, which is the one the head was
# fitted through; the `olmo` template in dataset_transformation.py injects
# function-calling boilerplate into the system message, which would prompt the
# policy in one context and score it in another. That template already begins with
# {{ bos_token }}, so --add_bos raises rather than being merely redundant.
set -euo pipefail

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$HOME/venv_test}
EXP=${EXP:-pedagogy_olmo7b}

export HF_HOME=$SCRATCH/hf-cache
export HF_HUB_CACHE=$SCRATCH/hf-cache/hub
export XDG_CACHE_HOME=$SCRATCH/xdg-cache
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor-cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=$REPO
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

source "$VENV/bin/activate"
cd "$REPO"
mkdir -p logs output

# 6 learners + 2 vLLM engines fills the node, which is the split the 8B
# single-node script uses. The reward runs inside the vLLM actors, so the two
# engines each carry a copy of the encoder.
python -u open_instruct/grpo_fast.py \
    --exp_name "$EXP" \
    --model_name_or_path allenai/OLMo-2-1124-7B-Instruct \
    --tokenizer_name_or_path allenai/OLMo-2-1124-7B-Instruct \
    --use_slow_tokenizer False \
    --dataset_mixer_list data/rl/train.jsonl 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list data/rl/eval.jsonl 1.0 \
    --dataset_mixer_eval_list_splits train \
    --reward_plugins projects/pedagogy_rm/plugin.py \
    --group_scorer "pedagogy:head=data/head.npz" \
    --group_reward_mode replace \
    --apply_verifiable_reward True \
    --max_prompt_token_length 1024 \
    --response_length 512 \
    --pack_length 2048 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 8 \
    --temperature 1.0 \
    --beta 0.02 \
    --learning_rate 3e-7 \
    --lr_scheduler_type constant_with_warmup \
    --warm_up_steps 10 \
    --total_episodes 100000 \
    --deepspeed_stage 3 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --num_learners_per_node 6 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.55 \
    --local_eval_every 10 \
    --save_freq 50 \
    --output_dir "output/$EXP" \
    --seed 1

echo "done $(date -Is)"
