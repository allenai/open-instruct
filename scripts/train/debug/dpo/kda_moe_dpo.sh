#!/bin/bash
# DPO on the OLMoE3 latent-KDA 1.2B MoE, starting from the #1854 think SFT
# checkpoint. Tracked in https://github.com/allenai/open-instruct/issues/1857.
# Sibling of scripts/train/debug/oc_sft_olmoe3_kda_think.sh, which produced the
# checkpoint this reads.
#
#   ./scripts/train/debug/dpo/kda_moe_dpo.sh $IMAGE $MODE
#
# Modes:
#   cache_dataset   CPU-only tokenization of the preference mix. Run first: rank 0
#                   otherwise tokenizes while 7 B300s idle at a barrier.
#   cache_logprobs  Reference log-probabilities for the whole mix, forward-only,
#                   1x8. dpo_norm needs a reference model, and open-instruct caches
#                   its log-probs to disk instead of holding a second 18.5B replica
#                   in memory -- which is what makes DPO fit here at all.
#   smoke           30 steps, 1x8. The memory gate (see below).
#   train           1 epoch over the full mix.
#   dense_control   The same code on a dense model that DPO is known to work on.
#                   Run this BEFORE reading any MoE result. DPO has not reproduced
#                   strong gains on the hybrid models trained so far, so a flat MoE
#                   result is a live possibility -- and without a reference curve
#                   from a model where DPO does work, "our code is wrong" and "DPO
#                   does not move this architecture" are indistinguishable. It goes
#                   through DPOTrainModule, which shares its objective with the MoE
#                   module via DPOObjectiveMixin, so a rising margin here clears the
#                   loss, the metrics and their reduction in one run.
#
# Why the settings are what they are:
#
# * The reference cache is keyed by compute_reference_cache_hash, which covers the
#   model, the dataset config (so SEQ and the mix), loss_type, packing,
#   concatenated_forward and MAX_SAMPLES. It does NOT cover the learning rate or
#   beta, so one cache serves an entire LR sweep -- but only while MAX_SAMPLES is
#   held fixed, which is the easy way to lose it, since sweeps run on subsets. A
#   miss is silent: the training job just rebuilds the cache on 8 GPUs.
#   It lives on weka (REFERENCE_LOGPROBS_CACHE_PATH), so it does survive between
#   Beaker jobs.
#
# * SEQ defaults to 16384, Olmo 3's think DPO length, not the SFT run's 32768.
#   DPO packs one preference pair per sequence, so the length is set by the data,
#   not by the base -- and every DPO micro-batch carries a chosen AND a rejected
#   sequence, so the microbatch is 2 x SEQ tokens where SFT's was 1 x.
#
# * THE MEMORY GATE IS NOT THE SAME QUESTION THE SFT GATE ANSWERED. DPOLMHead
#   replaces the LM head and computes `logits = self.w_out(h)` explicitly, so the
#   checkpoint's `loss_implementation: fused_linear` -- which exists precisely to
#   avoid materializing logits -- does not apply on this path. At SEQ=16384 and
#   vocab 100,352 that is a 3.3 GB bf16 logits tensor per sequence, twice per
#   instance, on top of the 165.3 GiB the SFT run reserved at 32768. Whether that
#   fits is unmeasured; the smoke exists to measure it. If it OOMs, the ladder is
#   SEQ=8192 first (halves both the activations and the logits), then
#   ACT_CKPT_MODE=selected_modules.
#
# * SEQ 16384 IS SHORTER THAN THE BASE'S NATIVE 65536, AND THAT IS A MEASURED RISK.
#   SFT on this family at seq 32768 destroyed length extrapolation: RULER at 65536 fell
#   to 0.0883 against the base's 0.3751, while improving ~+28pp at every length inside
#   the training window (#1854 / #1859). DPO at 16384 is shorter still. Nothing here
#   fixes that, and a short-context eval battery cannot see it -- so run RULER at the
#   native length on the DPO output, not just the in-window tasks, before calling any
#   DPO delta a win.
#
# * TERMINATION IS A FIRST-CLASS METRIC ON THIS FAMILY, NOT A FOOTNOTE. ~41% of
#   open-ended prompts run to the token cap after SFT (the base is 97%). DPO changes
#   response-length behaviour by construction, so measure the non-termination rate and
#   mean response length on the DPO output. Two practical consequences: an uncapped eval
#   task blows up ~15x in GPU-hours on this family, so check max_tokens on every task
#   before launching; and concision is a stated Olmo 3.5 goal, so a length regression is
#   a real result rather than noise.
#
# * PACKING is off. The DPO train module supports it only with a flash backend
#   for intra-document masking, and 16 of the 20 blocks here are KDA linear
#   attention rather than flash. Turning it on without checking whether KDA
#   honours doc_lens would silently let sequences attend across pair boundaries.
#
# * LR defaults to 8e-8, Olmo 3's 7B think DPO value. It is not validated at 1.2B
#   active -- sweep 8e-8 / 3e-7 / 1e-6 on a subset (MAX_SAMPLES) before the full
#   run, per the issue.
#
# * Tokenizer and chat template are held identical to the SFT run. docs/olmo3.md
#   recommends olmo-3.2-tokenizer-think-dev for post-SFT stages, but changing the
#   prompt format at the DPO boundary would confound a pipeline validation with a
#   template change. Note that --chat_template_name olmo123 is NOT in
#   CHAT_TEMPLATES: get_tokenizer_tulu_v2_2 falls back to the tokenizer repo's own
#   template, which is the one SFT trained on. The flag is inert; the tokenizer
#   repo is what matters.
#
# * No --preemptible on holmes. It sets minRuntime 0, which makes the job
#   backfill-only on a strict-priority cluster, so it never schedules when full.
#   An allocated job gets an 8 h min-runtime shield instead, hence the timeout.
set -euo pipefail

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
MODE="${2:-smoke}"

# The #1854 SFT checkpoint: 0.5 epoch of Dolci-Think on the long-context KDA base.
MODEL="${MODEL:-/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/7u2rne45/step11768}"
# The architecture the SFT run trained, including recompute_each_block. dpo.py
# needs it because an olmo-core checkpoint carries no HF config to build from.
CONFIG_NAME="${CONFIG_NAME:-scripts/train/debug/kda_lc_sft.json}"
TOKENIZER="${TOKENIZER:-allenai/olmo-3-tokenizer-instruct-dev}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-olmo123}"
MIXER="${MIXER:-allenai/Dolci-Think-DPO-7B 1.0}"

SEQ="${SEQ:-16384}"
LR="${LR:-8e-8}"
BETA="${BETA:-5}"
LOSS_TYPE="${LOSS_TYPE:-dpo_norm}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
ACT_MEM_BUDGET="${ACT_MEM_BUDGET:-1}"
ACT_CKPT_MODE="${ACT_CKPT_MODE:-budget}"
# 1, not dpo.py's default 4. The reference pass at 4x puts 8 sequences (131k tokens) through
# fla's chunked KDA kernels in a single call; two separate B300 nodes took unrecoverable Xid
# 31/43 faults ~90s in, at batch 0 of 63, and were cordoned. SFT ran for hours on the same
# image with 1 sequence per call. Raise this only with evidence that the kernels tolerate it.
REF_CACHE_MULT="${REF_CACHE_MULT:-1}"
# The smoke defaults to a subset because the reference log-probability pass is
# forward-only over the WHOLE dataset before training starts. On the full 149,986-pair
# mix that pass, not the 30 training steps, is what a smoke would actually pay for.
MAX_SAMPLES="${MAX_SAMPLES:-}"
if [[ "${2:-smoke}" == "smoke" && -z "$MAX_SAMPLES" ]]; then
    MAX_SAMPLES=2000
fi

CLUSTER="${CLUSTER:-ai2/holmes}"
WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"
PRIORITY="${PRIORITY:-high}"
NNODES="${NNODES:-1}"
NPROC="${NPROC:-8}"
JOB_TIMEOUT="${JOB_TIMEOUT:-2h}"
MAX_RETRIES="${MAX_RETRIES:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/kda-moe-dpo-$(date +%s)}"
PY="${PY:-uv run python}"

EXP_NAME="kda-moe-dpo-${MODE}-lr${LR}-seq${SEQ}-$(date +%s)"

# Arguments shared by every mode. Kept in one place so the reference cache built by
# cache_logprobs hashes identically to the one the training run looks for.
common_args=(
    --exp_name "$EXP_NAME"
    --model_name_or_path "$MODEL"
    --config_name "$CONFIG_NAME"
    --tokenizer_name_or_path "$TOKENIZER"
    --chat_template_name "$CHAT_TEMPLATE"
    --mixer_list $MIXER
    --max_seq_length "$SEQ"
    --loss_type "$LOSS_TYPE"
    --beta "$BETA"
    --learning_rate "$LR"
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --lr_scheduler_type linear
    --warmup_ratio 0.1
    --weight_decay 0.0
    --max_grad_norm 1.0
    --num_epochs 1
    --attn_implementation flash_2
    --activation_checkpointing_mode "$ACT_CKPT_MODE"
    --activation_memory_budget "$ACT_MEM_BUDGET"
    --reference_cache_batch_multiplier "$REF_CACHE_MULT"
    --output_dir "$OUTPUT_DIR"
    --logging_steps 1
    --seed 123
    --push_to_hub false
    --try_launch_beaker_eval_jobs false
)
if [[ -n "$MAX_SAMPLES" ]]; then
    common_args+=(--max_train_samples "$MAX_SAMPLES")
fi

case "$MODE" in
    cache_dataset)
        DESC="KDA MoE DPO: tokenize $MIXER at $SEQ"
        $PY mason.py \
            --cluster ai2/saturn ai2/neptune ai2/ceres ai2/jupiter \
            --workspace "$WORKSPACE" --priority "$PRIORITY" \
            --image "$BEAKER_IMAGE" --description "$DESC" --pure_docker_mode \
            --timeout "${TOKENIZE_TIMEOUT:-4h}" \
            --num_nodes 1 --gpus 0 --non_resumable --no_auto_dataset_cache \
            -- uv run python open_instruct/dpo.py "${common_args[@]}" --cache_dataset_only
        ;;
    cache_logprobs|smoke|train)
        case "$MODE" in
            # Forward-only over the whole mix, so it needs no step cap and no writes.
            cache_logprobs) EXTRA=(--cache_logprobs_only); DESC="KDA MoE DPO: reference logprobs at $SEQ" ;;
            # Checkpointing is left off: each write is 207 GB and synchronous (the DDP
            # train module rejects async saves), which a memory gate does not need.
            smoke)          EXTRA=(--max_train_steps "${STEPS:-30}"); DESC="KDA MoE DPO smoke: ${STEPS:-30} steps at $SEQ" ;;
            train)          EXTRA=(--checkpointing_steps "${CKPT_STEPS:-500}" --with_tracking); DESC="KDA MoE DPO: 1 epoch, lr $LR, seq $SEQ" ;;
        esac
        if (( NNODES > 1 )); then
            RDZV_FLAGS="--node_rank=\$BEAKER_REPLICA_RANK --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME --master_port=29400"
        else
            # torchrun defaults to 29500; two single-node jobs on one host collide there
            # during rendezvous, before any model is built, which yields no memory
            # information and reads like a real failure.
            RDZV_FLAGS="--master_port=${MASTER_PORT:-$(( 29500 + RANDOM % 400 ))}"
        fi
        $PY mason.py \
            --cluster "$CLUSTER" \
            --workspace "$WORKSPACE" --priority "$PRIORITY" \
            --image "$BEAKER_IMAGE" --description "$DESC" --pure_docker_mode \
            --timeout "$JOB_TIMEOUT" --max_retries "$MAX_RETRIES" \
            --num_nodes "$NNODES" --gpus "$NPROC" \
            --non_resumable --no_auto_dataset_cache \
            --env OLMO_SHARED_FS=1 \
            -- torchrun --nnodes="$NNODES" $RDZV_FLAGS --nproc_per_node="$NPROC" \
            open_instruct/dpo.py "${common_args[@]}" "${EXTRA[@]}"
        ;;
    dense_control)
        # Settings follow scripts/train/debug/dpo/single_gpu.sh (the known-good dense
        # path) rather than this script's: OLMo-2-0425-1B, its LR and chat template,
        # its preference mix. Only the sample count is raised, because 100 pairs is a
        # smoke test rather than a curve, and the cluster is holmes -- the cuda13 image
        # this branch builds needs a CUDA 13 driver, which saturn/jupiter do not have.
        DESC="Dense DPO control for the KDA MoE arm"
        $PY mason.py \
            --cluster ai2/holmes \
            --workspace "$WORKSPACE" --priority "$PRIORITY" \
            --image "$BEAKER_IMAGE" --description "$DESC" --pure_docker_mode \
            --timeout "${JOB_TIMEOUT:-2h}" --max_retries "$MAX_RETRIES" \
            --num_nodes 1 --gpus 1 --non_resumable --no_auto_dataset_cache \
            -- torchrun --nproc_per_node=1 --master_port="${MASTER_PORT:-$(( 29500 + RANDOM % 400 ))}" \
            open_instruct/dpo.py \
            --exp_name "dense-dpo-control-$(date +%s)" \
            --model_name_or_path allenai/OLMo-2-0425-1B \
            --tokenizer_name_or_path allenai/OLMo-2-0425-1B \
            --chat_template_name olmo \
            --mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b "${CONTROL_SAMPLES:-20000}" \
            --max_seq_length "${CONTROL_SEQ:-2048}" \
            --loss_type "$LOSS_TYPE" --beta "$BETA" \
            --learning_rate "${CONTROL_LR:-5e-7}" \
            --per_device_train_batch_size 2 \
            --gradient_accumulation_steps 8 \
            --lr_scheduler_type linear --warmup_ratio 0.1 --weight_decay 0.0 \
            --num_epochs 1 \
            --output_dir "$OUTPUT_DIR-dense-control" \
            --logging_steps 1 --seed 123 \
            --push_to_hub false --try_launch_beaker_eval_jobs false --try_auto_save_to_beaker false \
            --with_tracking
        ;;
    *)
        echo "Unknown mode: $MODE (expected cache_dataset, cache_logprobs, smoke, train or dense_control)" >&2
        exit 1
        ;;
esac
