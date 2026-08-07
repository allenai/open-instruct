#!/bin/bash
# Olmo-3-7B SFT on the full Dolci-Instruct-SFT mixture, capped at 1000 steps.
#
# This is the first real training run through the SFT label-derivation fix in
# https://github.com/allenai/open-instruct/issues/1800 -- Olmo-3's eos is <|endoftext|>, not
# <|im_end|>, so `olmo_thinker_no_think_sft_tokenization` takes the token-count fallback.
#
# Note: `--add_bos` must NOT be passed here. For an OLMo tokenizer with an "olmo*" chat
# template, dataset_transformation asserts add_bos is off (it is only required with older,
# non-olmo templates such as `tulu`).
#
# Ceres/Saturn rather than Jupiter: Jupiter uses Strict Priority scheduling, where a
# --preemptible job only ever gets backfill capacity and can queue for hours on a busy
# cluster. Ceres (H100 80GB) and Saturn (A100 80GB) are both Eager-scheduled and sized for
# exactly this kind of small distributed job.
#
# TWO GPUs. A small slot request still schedules quickly, but unlike world_size=1 it keeps
# FSDP sharding. A single-GPU attempt (grad_accum 64) hit `nan loss encountered at step 6`
# with LR still at 3.3e-6 -- far too small to diverge, and the 8-GPU run passed the same step
# on the same data. The NaN is in the single-rank path, not the data or the labels.
#
# Needs B300 (288GB) unless sharding across enough ranks: a 7B full-finetune with AdamW is
# ~112GB steady state (14GB bf16 params + 14GB grads + 84GB fp32 optimizer state and master
# weights) plus activations, which does not fit one 80GB H100.
#
# Global batch is held at 64 sequences (2 GPUs * per_device 1 * grad_accum 32), identical to
# every other configuration tried, so the learning rate below stays valid unchanged. 1000
# steps is ~64k sequences: a partial pass over Dolci, not a full epoch.
# LR 5e-6 with gradient clipping. Do not copy the reference recipe's 8e-5: that run
# (scripts/train/olmo3/7b_instruct_sft.sh) *resumes from an olmo-core checkpoint with
# optimizer state*, while this starts cold from the HF base, so its LR does not transfer even
# after scaling for batch size. Two runs diverged once LR hit peak, both recoverable only by
# lowering it: 8e-5 -> CE 0.95@10 to 8.29@20 (wandb c7odd0hp); 2e-5 -> CE 0.87@10 to 2.69@30
# to 8.70@60 (wandb hn9nfjfh).
#
# max_grad_norm defaults to None (no clipping at all), which lets a single bad batch move the
# weights arbitrarily far. Clipping at 1.0 is the standard guard and costs nothing.

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
# Cluster is overridable so the same recipe can chase whichever pool has capacity.
# NOTE: ai2/holmes is B300 and only accepts CUDA 13 images -- build with
# `build_image_and_launch.sh --cuda-version 13` when targeting it (see PR #1758).
CLUSTER="${2:-ai2/ceres ai2/saturn}"

echo "Using Beaker image: $BEAKER_IMAGE"
echo "Targeting cluster(s): $CLUSTER"

uv run python mason.py \
    --cluster $CLUSTER \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Olmo-3-7B SFT, Dolci-Instruct-SFT, 1000 steps, seq4096, 1x8 (issue 1800 fix)" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --gpus 2 \
    --non_resumable \
    --no-host-networking \
    --no_auto_dataset_cache \
    -- torchrun \
    --nproc_per_node=2 \
    open_instruct/olmo_core_finetune.py \
    --model_name_or_path allenai/Olmo-3-1025-7B \
    --chat_template_name olmo_thinker_no_think_sft_tokenization \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-6 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.03 \
    --num_epochs 1 \
    --max_train_steps 1000 \
    --ephemeral_save_interval 250 \
    --logging_steps 1 \
    --mixer_list allenai/Dolci-Instruct-SFT 1.0 \
    --seed 123 \
    --with_tracking \
    --output_dir \$CHECKPOINT_OUTPUT_DIR
