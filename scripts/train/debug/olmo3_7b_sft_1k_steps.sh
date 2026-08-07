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
# Single node: 8 GPUs schedule far more easily than 2x8, and this is a correctness run --
# throughput does not matter. Global batch is held at 64 sequences (8 GPUs * per_device 1 *
# grad_accum 8) so the batch/LR relationship matches the reference recipe rather than being
# silently halved by using fewer GPUs. 1000 steps is ~64k sequences: a partial pass over
# Dolci, not a full epoch.
# LR matches the Olmo 3 reference 7B SFT recipe (scripts/train/olmo3/7b_instruct_sft.sh).

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Olmo-3-7B SFT, Dolci-Instruct-SFT, 1000 steps, seq4096, 1x8 (issue 1800 fix)" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --gpus 8 \
    --non_resumable \
    --no-host-networking \
    --no_auto_dataset_cache \
    -- torchrun \
    --nproc_per_node=8 \
    open_instruct/olmo_core_finetune.py \
    --model_name_or_path allenai/Olmo-3-1025-7B \
    --chat_template_name olmo_thinker_no_think_sft_tokenization \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 8e-5 \
    --warmup_ratio 0.03 \
    --num_epochs 1 \
    --max_train_steps 1000 \
    --ephemeral_save_interval 250 \
    --logging_steps 1 \
    --mixer_list allenai/Dolci-Instruct-SFT 1.0 \
    --seed 123 \
    --with_tracking \
    --output_dir \$CHECKPOINT_OUTPUT_DIR
