#!/bin/bash
# Wandb: https://wandb.ai/ai2-llm/jacobm-7B-sft/runs/zfn667tc
# Beaker: https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01KA38QQENAC3S8GEKVYHYT2MV
# Commit: 9e97471057d7046f0ae7315e0225d117b54186f9

CHECKPOINT=gs://ai2-llm/jacobm/checkpoints/olmo3-7b-reasoning-sft-final/model_and_optim
LR=8e-5
python src/scripts/train/sft/OLMo-sft.py train \
    olmo3-7b-instruct-SFT-1114-fix-${LR}-seed_543210 \
        $CHECKPOINT \
        ai2/augusta \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=$LR \
    --seq_len=32768 \
    --launch.num_gpus=8 \
    --num_nodes=4 \
    --global_batch_size=1048576 \
    --model_name=olmo3-7b \
    --budget=ai2/oe-adapt \
    --workspace=ai2/olmo-instruct \
    --dataset_path=gs://ai2-llm/jacobm/data/sft/rl-sft-32k/olmo3-32b-instruct-sft-1114 \
    --launch.priority=urgent
