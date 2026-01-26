#!/bin/bash
# Wandb 1: https://wandb.ai/ai2-llm/saumyam-7b-sft/runs/4yx5d5bk
# Wandb 2: https://wandb.ai/ai2-llm/saumyam-7b-sft/runs/3t0hzqap
# Beaker: https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01K5FBWEXJV35P4J9H5FJQ7WE6
# Commit: 38f66526c9d1ba6b97269ebfb429749a5feb528f

CHECKPOINT=/weka/oe-training-default/ai2-llm/checkpoints/tylerr/long-context/olmo25_7b_lc_64k_6T_M100B_round5-sparkle_6634-pre_s2pdf_gzip2080_cweN-yake-all-olmo_packing_yarn-fullonly_50B-fb13a737/step11921/
LR=5e-5
python src/scripts/train/sft/OLMo2-7B-sft.py train \
    olmo2.5-6T-LC-sigma-reasoning-mix-decontam-v2-special-tokens-v3-think-FIX \
        $CHECKPOINT \
        ai2/jupiter-cirrascale-2 \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=$LR \
    --seq_len=32768 \
    --launch.num_gpus=8 \
    --num_nodes=8 \
    --global_batch_size=1048576 \
    --model_name=olmo2.5-7b-yarn-fullonly \
    --budget=ai2/oe-adapt \
    --workspace=ai2/olmo-instruct \
    --dataset_path=/weka/oe-training-default/ai2-llm/jacobm/data/sft/rl-sft-32k/reasoning-mix-decontam-v2-special-tokens-v3-think-FIX \
    --launch.priority=urgent
