#!/bin/bash
# Wandb 1e-4: https://wandb.ai/ai2-llm/saumyam-7B-sft/runs/gn5kre41
# Wandb 5e-5: https://wandb.ai/ai2-llm/saumyam-7B-sft/runs/twcn6j46
# Beaker 1e-4: https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01K9HRTZQGZJV22MPDZDK6KG56
# Beaker 5e-5: https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01K9HWJQS5N23K3W02YHKEG6BS
# Commit: 79a184cf70d83df6bcb7fe6f5fadffbc717b6ce5

CHECKPOINT=gs://ai2-llm/checkpoints/stego32-longcontext-run-3/step11921+11000+10000-nooptim
LR=1e-4
python src/scripts/train/sft/OLMo-sft.py train \
    olmo3-32b-SFT-${LR} \
        $CHECKPOINT \
        ai2/augusta \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=$LR \
    --seq_len=32768 \
    --launch.num_gpus=8 \
    --num_nodes=32 \
    --global_batch_size=4194304 \
    --launch.use_hostname_constraints=True \
    --launch.num_execution_units=1 \
    --model_name=olmo3-32b \
    --budget=ai2/oe-adapt \
    --workspace=ai2/olmo-instruct \
    --dataset_path=gs://ai2-llm/jacobm/data/sft/rl-sft-32k/olmo3-32b-thinking-sft \
    --launch.priority=urgent
