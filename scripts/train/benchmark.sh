# ------------------------------------------------------------
# This script is used to keep track of the commands used to launch
# the experiments in https://wandb.ai/ai2-llm/open_instruct_public.


# ------------------------------------------------------------
# Tulu3

# 8 nodes
python update_command_args.py scripts/train/tulu3/finetune_8b.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --seed 1 \
    --image costah/open_instruct_dev0327_4 | uv run bash

# 4 nodes
python update_command_args.py scripts/train/tulu3/dpo_8b.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --seed 1 \
    --image costah/open_instruct_dev0320_11 | uv run bash


# 2 nodes
python update_command_args.py scripts/train/tulu3/reward_modeling_8b.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority high \
    --seed 1 \
    --image costah/open_instruct_dev0320_11 | uv run bash

# 2 nodes
python update_command_args.py scripts/train/tulu3/grpo_8b.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --seed 40 \
    --image costah/open_instruct_dev0320_11 | uv run bash


# 2 nodes
python update_command_args.py scripts/train/tulu3/ppo_8b.sh \
    --wandb_project_name open_instruct_public \
    --priority high \
    --image costah/open_instruct_dev_uv13 | uv run bash

# 2 nodes
python update_command_args.py scripts/train/tulu3/grpo_fast_8b.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --seed 40 \
    --image costah/open_instruct_dev0320_11 | uv run bash

# 1 node
python update_command_args.py scripts/train/tulu3/grpo_fast_8b_single_node.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --seed 40 \
    --image costah/open_instruct_dev0320_11 | uv run bash


# ------------------------------------------------------------
# Qwen

# 2 nodes
python update_command_args.py scripts/train/qwen/grpo_fast_7b.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority urgent | uv run bash

# 4 nodes
python update_command_args.py scripts/train/qwen/grpo_fast_7b_orz.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --image costah/open_instruct_dev_0405 \
    --priority high | uv run bash

# 3 nodes
python update_command_args.py scripts/train/qwen/ppo_fast_7b_orz.sh \
    --cluster ai2/augusta \
    --priority high \
    --workspace ai2/scaling-rl \
    --wandb_project_name open_instruct_public \
    --image costah/open_instruct_dev_0427_ppo_17 | uv run bash

# 2 nodes
python update_command_args.py scripts/train/qwen/grpo_7b.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority high \
    --workspace ai2/scaling-rl \
    --image costah/open_instruct_dev_0410_ww_1 | uv run bash

# 1 node
python update_command_args.py scripts/train/qwen/grpo_fast_3b_single_node.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0320_11 | uv run bash


# 8 nodes
python update_command_args.py scripts/train/qwen/grpo_fast_32b.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority high | uv run bash

# ------------------------------------------------------------
# Llama3

# 4 nodes
python update_command_args.py scripts/train/llama3/grpo_fast_7b_math.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority high | uv run bash

# ------------------------------------------------------------
# Olmo2

# 8 nodes
python update_command_args.py scripts/train/olmo2/finetune_13b.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority high \
    --image costah/open_instruct_dev_uv12 | uv run bash

# 8 nodes
python update_command_args.py scripts/train/olmo2/finetune_7b.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0320_11 | uv run bash

# 8 nodes
python update_command_args.py scripts/train/olmo2/finetune_32b.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority high \
    --image costah/open_instruct_dev0320_11 | uv run bash

# 4 nodes
python update_command_args.py scripts/train/olmo2/dpo_7b.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0320_11 | uv run bash

# 4 nodes
python update_command_args.py scripts/train/olmo2/dpo_13b.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0320_11 | uv run bash

# 2 nodes
python update_command_args.py scripts/train/olmo2/grpo_fast_7b_zero.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority urgent \
    --image costah/open_instruct_dev0327_4 | uv run bash

# 2 nodes
python update_command_args.py scripts/train/olmo2/grpo_fast_13b_zero.sh \
    --cluster ai2/augusta \
    --wandb_project_name open_instruct_public \
    --priority urgent \
    --image costah/open_instruct_dev0327_4 | uv run bash
