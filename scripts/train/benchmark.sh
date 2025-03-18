# ------------------------------------------------------------
# Tulu3

# 8 nodes
python launch.py scripts/train/tulu3/finetune_8b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0316 | uv run bash

# 4 nodes
python launch.py scripts/train/tulu3/dpo_8b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0316_7 | uv run bash

# 2 nodes
python launch.py scripts/train/tulu3/grpo_8b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0316 | uv run bash

# 2 nodes
python launch.py scripts/train/tulu3/grpo_fast_8b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0316_16 | uv run bash

# ------------------------------------------------------------
# Qwen

# 2 nodes
python launch.py scripts/train/qwen/grpo_fast_7b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0316_16 | uv run bash


# 2 nodes
python launch.py scripts/train/qwen/grpo_7b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0316_16 | uv run bash

# ------------------------------------------------------------
# Olmo2

# 8 nodes
python launch.py scripts/train/olmo2/finetune_13b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0316_17 | uv run bash

# 8 nodes
python launch.py scripts/train/olmo2/finetune_7b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0316_17 | uv run bash

# 4 nodes
python launch.py scripts/train/olmo2/dpo_7b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0316_17 | uv run bash

# 4 nodes
python launch.py scripts/train/olmo2/dpo_13b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0316_17 | uv run bash





python launch.py scripts/train/tulu3/grpo_fast_8b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --seed 3 \
    --beta 0.00 \
    --masked_mean_axis 1 \
    --image costah/open_instruct_dev0316_16 | uv run bash

# (not part of the benchmark) 2 nodes
python launch.py scripts/train/tulu3/grpo_fast_8b.sh \
    --cluster ai2/augusta-google-1 \
    --priority normal \
    --wandb_project_name open_instruct_public \
    --exp_name tulu3.1_8b_grpo_fast_axis_1_mean \
    --masked_mean_axis 1 \
    --image costah/open_instruct_dev0316_15 | uv run bash


# ------------------------------------------------------------
# single gpu experiments
python launch.py scripts/train/qwen/grpo_fast_3b_single_node.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0316_10 | uv run bash


python launch.py scripts/train/tulu3/grpo_fast_8b_single_node.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0316_11 | uv run bash
