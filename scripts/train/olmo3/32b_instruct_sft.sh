#!/bin/bash
# Wandb: https://wandb.ai/ai2-llm/jacobm-7B-sft/runs/en7w8mj1
# Beaker: https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01KA4TA2T20EPNG1ADPBQ3A3RZ
# Commit: abfa4eaaf6dfa2b77b1a9586ccca31013fc3e4ea

CHECKPOINT=gs://ai2-llm/jacobm/olmo3-32b-reasoning-sft/model_and_optim
LR=8e-5
python src/scripts/train/sft/OLMo-sft.py train \
    olmo3-32b-instruct-SFT-1114-fix-${LR}-seed_33333 \
        $CHECKPOINT \
        ai2/augusta \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=$LR \
    --seq_len=32768 \
    --launch.num_gpus=8 \
    --num_nodes=8 \
    --global_batch_size=4194304 \
    --model_name=olmo3-32b \
    --budget=ai2/oe-adapt \
    --workspace=ai2/olmo-instruct \
    --dataset_path=gs://ai2-llm/jacobm/data/sft/rl-sft-32k/olmo3-32b-instruct-sft-1114 \
    --launch.priority=urgent
