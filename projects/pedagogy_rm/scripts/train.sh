#!/bin/bash
#
# GRPO on OLMo-2-7B-Instruct, rewarded by the hidden-state probe.
#
# The invocation lives here rather than in the sbatch file because two very different
# things have to run it: a Slurm allocation on the MIT cluster, and an AWS Batch
# container on the eduLLM platform, which execs a command string and knows nothing
# about modules or venvs. Everything environment-specific is the caller's job; this
# file is the run itself.
#
# TWO TUNING MODES, AND THE CHOICE IS ABOUT MEMORY RATHER THAN ABOUT METHOD.
#
#   MODE=full  full-weight, DeepSpeed ZeRO-3 sharded across LEARNERS cards.
#   MODE=lora  low-rank adapters on a frozen base, ZeRO-2, fits on one card.
#
# What does not fit on one card is the optimizer, not the model. Adam holds fp32
# momentum, variance and master weights, so 7.3B parameters cost about 88GB of state
# before a single activation - against 14.6GB for the bf16 weights themselves. Stage 3
# divides that 88GB by the number of learners, and OFFLOAD=1 moves it to host RAM
# instead, which is what lets four 48GB cards do full-weight training at all.
#
# WHY FULL WEIGHT IS SAFE HERE, given that the reward is a linear head on one
# checkpoint's activations. The head only keeps its meaning while those activations
# hold still, and full-weight training moves the policy's. It stays valid because the
# scorer does not read the policy: it loads its own frozen copy of the encoder inside
# the vLLM actor, from the checkpoint named in data/head.npz. So the policy cannot
# raise its reward by drifting its representations - the space it is read in is not
# one it owns. What remains is a distribution risk rather than a definitional one: a
# policy trained far enough can write text unlike anything the probe was validated on,
# where its predictions mean less. That is what --beta and the held-out anchor eval
# are for, and it is the reason LoRA is still the better choice on one GPU rather than
# merely the affordable one.
#
# NO --chat_template_name AND NO --add_bos, both deliberate. Leaving the template unset
# makes open-instruct use the tokenizer's own, which is the one the head was fitted
# through; the `olmo` template in dataset_transformation.py injects function-calling
# boilerplate into the system message, which would prompt the policy in one context and
# score it in another. That template already begins with {{ bos_token }}, so --add_bos
# raises rather than being merely redundant.
set -euo pipefail

MODE=${MODE:-lora}
EXP=${EXP:-pedagogy_olmo7b}

# The encoder is deliberately not configurable: it is pinned inside data/head.npz, so
# every run - including a smoke run behind a 1B policy - scores through the same 7B the
# head was fitted on.
POLICY=${POLICY:-allenai/OLMo-2-1124-7B-Instruct}

GPUS=${GPUS:-1}
LEARNERS=${LEARNERS:-1}
ENGINES=${ENGINES:-1}
TP=${TP:-1}
OFFLOAD=${OFFLOAD:-0}

EPISODES=${EPISODES:-100000}
PROMPTS=${PROMPTS:-32}
SAMPLES=${SAMPLES:-8}
MICRO_BATCH=${MICRO_BATCH:-1}
# Recomputing activations in the backward instead of holding them. On by default because
# it is what makes a 7B fit beside vLLM and the scorer's encoder on one 80GB card; a card
# with room to spare should turn it off, since the memory it saves costs a second forward
# pass through every checkpointed block. It changes speed and memory, not the update.
GRAD_CKPT=${GRAD_CKPT:-1}
EVAL_EVERY=${EVAL_EVERY:-10}
SAVE_FREQ=${SAVE_FREQ:-50}
BETA=${BETA:-0.02}
SEED=${SEED:-1}
OUTPUT_DIR=${OUTPUT_DIR:-output/$EXP}

LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}

# Where trainer state goes so a killed run resumes instead of restarting. Empty disables
# it, which is right for a smoke run and wrong for anything long: see the flags below.
CKPT_ROOT=${CKPT_ROOT:-}

# Which registered scorer computes the reward. `pedagogy` averages the four signed
# dimensions on their raw 1-3 scales; `pedagogy_z` z-scores each one inside the group first,
# so they contribute equally rather than in proportion to their spread. plugin.py argues
# both sides. They are two arms of one experiment, not a setting with a right answer.
SCORER=${SCORER:-pedagogy}

CACHE_ROOT=${CACHE_ROOT:-${SCRATCH:-/tmp}}
export HF_HOME=${HF_HOME:-$CACHE_ROOT/hf-cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$CACHE_ROOT/hf-cache/hub}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$CACHE_ROOT/xdg-cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$CACHE_ROOT/triton-cache}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/inductor-cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${PYTHONPATH:-$PWD}

# Ray places LEARNERS learner actors and ENGINES vLLM engines of TP cards each, so the
# three have to add up to the machine. They are checked here rather than left to fail
# somewhere inside Ray placement, where the error is a group that never becomes ready
# rather than a sentence about arithmetic. The platform runs its own version of this
# check before a job is ever submitted; this one is for the clusters that do not.
NEEDED=$((LEARNERS + ENGINES * TP))
COLOCATE=0
if [ "$NEEDED" -gt "$GPUS" ]; then
    if [ "$GPUS" -eq 1 ] && [ "$LEARNERS" -eq 1 ] && [ "$ENGINES" -eq 1 ] && [ "$TP" -eq 1 ]; then
        # The one oversubscription that works: --single_gpu_mode gives the learner 0.48
        # of the card and vLLM 0.5, and the weight sync drops from NCCL to CUDA IPC
        # because both live on the same device.
        COLOCATE=1
    else
        echo "LEARNERS=$LEARNERS + ENGINES=$ENGINES x TP=$TP needs $NEEDED cards, have GPUS=$GPUS" >&2
        exit 2
    fi
elif [ "$NEEDED" -lt "$GPUS" ]; then
    echo "LEARNERS=$LEARNERS + ENGINES=$ENGINES x TP=$TP uses $NEEDED of $GPUS cards; $((GPUS - NEEDED)) would idle" >&2
    exit 2
fi

tuning=()
if [ "$MODE" = full ]; then
    # 3e-7 rather than the 1e-5 LoRA wants: this update lands on the weights themselves
    # rather than on a low-rank adapter starting from zero.
    tuning+=(--learning_rate "${LR:-3e-7}" --deepspeed_stage 3)
    if [ "$OFFLOAD" = 1 ]; then
        # Adam on the host. Costs step time - the copy is over PCIe and the update is on
        # CPU - and buys back 88GB/LEARNERS of device memory, which is the difference
        # between fitting and not on any 48GB card.
        tuning+=(--deepspeed_offload_optimizer)
    fi
elif [ "$MODE" = lora ]; then
    # STAGE 2, NOT 3, and not merely because there is little left to shard. Stage 3
    # splits parameters across ranks, and merge_adapter has to see a whole base weight
    # to fold an adapter into it; grpo_fast raises rather than let that corrupt the send
    # to vLLM.
    tuning+=(
        --learning_rate "${LR:-1e-5}"
        --deepspeed_stage 2
        --use_peft
        --lora_r "$LORA_R"
        --lora_alpha "$LORA_ALPHA"
        --lora_dropout 0.0
    )
else
    echo "MODE must be full or lora, got '$MODE'" >&2
    exit 2
fi

[ "$COLOCATE" = 1 ] && tuning+=(--single_gpu_mode)

# --push_to_hub defaults to TRUE upstream, which publishes the trained policy to the
# Hub under whatever account the environment happens to be logged into. It also fails
# before training starts: setup_runtime_variables resolves the target repo by calling
# HfApi().whoami(), so on a cluster with HF_HUB_OFFLINE set the run dies in argument
# handling rather than at the end.
tuning+=(--push_to_hub False)

# Tracking is off unless a project is named. The eduLLM platform always names one - it
# is a required field on the submission form and the W&B key reaches the container
# through its execution role. On the MIT nodes there is no route to the internet, so the
# caller sets WANDB_MODE=offline and the run writes a local directory to sync later;
# without that the reward curve, which is the actual output of the experiment, exists
# only in a terminal.
if [ -n "${WANDB_PROJECT:-}" ]; then
    tuning+=(--with_tracking --wandb_project "$WANDB_PROJECT")
    # Passed explicitly wherever it is known, because leaving it unset makes
    # open-instruct call wandb.login() to check for an Ai2 team it will not find - a
    # network round trip that fails on an offline node before training starts.
    [ -n "${WANDB_ENTITY:-}" ] && tuning+=(--wandb_entity "$WANDB_ENTITY")
fi

# RESUMABLE STATE, WHICH ON A PREEMPTABLE PARTITION IS THE DIFFERENCE BETWEEN A REQUEUE
# COSTING MINUTES AND COSTING THE RUN. mit_preemptable really does preempt - a smoke run
# was taken at 2:13 and requeued - and Slurm restarts the script from the top, so without
# this the second attempt begins at step zero. grpo_fast resumes on its own when the
# directory holds state: it reads optimization_steps_done and continues from the step
# after. The frequency is tied to --save_freq because grpo_utils warns when they differ,
# and a warning about two frequencies is worth less than having one number to reason about.
if [ -n "$CKPT_ROOT" ]; then
    tuning+=(--checkpoint_state_dir "$CKPT_ROOT/$EXP" --checkpoint_state_freq "$SAVE_FREQ")
fi

# Tested against "0" explicitly rather than for emptiness, because "0" is a non-empty
# string and ${GRAD_CKPT:+...} would cheerfully add the flag it was meant to remove.
[ "$GRAD_CKPT" != "0" ] && tuning+=(--gradient_checkpointing)

# vLLM's share of a card it has to itself can be generous; a card it shares with a
# learner and with the scorer's frozen encoder cannot. The caller sets it, because only
# the caller knows the card.
VLLM_UTIL=${VLLM_UTIL:-$([ "$COLOCATE" = 1 ] && echo 0.30 || echo 0.55)}

echo "mode=$MODE policy=$POLICY gpus=$GPUS learners=$LEARNERS engines=${ENGINES}x${TP}" \
     "offload=$OFFLOAD colocate=$COLOCATE vllm_util=$VLLM_UTIL"

exec python -u open_instruct/grpo_fast.py \
    --exp_name "$EXP" \
    --model_name_or_path "$POLICY" \
    --tokenizer_name_or_path "$POLICY" \
    --use_slow_tokenizer False \
    --dataset_mixer_list data/rl/train.jsonl 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list data/rl/eval.jsonl 1.0 \
    --dataset_mixer_eval_list_splits train \
    --reward_plugins projects/pedagogy_rm/plugin.py \
    --group_scorer "$SCORER:head=data/head.npz" \
    --group_reward_mode replace \
    --apply_verifiable_reward True \
    --max_prompt_token_length 1024 \
    --response_length 512 \
    --pack_length 2048 \
    --num_unique_prompts_rollout "$PROMPTS" \
    --num_samples_per_prompt_rollout "$SAMPLES" \
    --temperature 1.0 \
    --beta "$BETA" \
    "${tuning[@]}" \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.03 \
    --total_episodes "$EPISODES" \
    --per_device_train_batch_size "$MICRO_BATCH" \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --num_learners_per_node "$LEARNERS" \
    --vllm_num_engines "$ENGINES" \
    --vllm_tensor_parallel_size "$TP" \
    --vllm_gpu_memory_utilization "$VLLM_UTIL" \
    --local_eval_every "$EVAL_EVERY" \
    --save_freq "$SAVE_FREQ" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED"
