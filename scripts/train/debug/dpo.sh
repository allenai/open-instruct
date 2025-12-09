#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

uv run python mason.py \
    --cluster ai2/saturn \
    --cluster ai2/jupiter \
    --description "Single GPU DPO run with OLMo-core, for debugging purposes." \
    --workspace ai2/tulu-thinker \
    --priority high \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --gpus 1 -- python open_instruct/dpo.py \
    --model_name_or_path allenai/OLMo-2-0425-1B \
    --tokenizer_name allenai/OLMo-2-0425-1B \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-07 \
    --warmup_ratio 0.1 \
    --num_epochs 1 \
    --output_dir output/dpo_olmo_debug/ \
    --dataset_mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 100 \
    --seed 123
