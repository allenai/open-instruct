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
    --set_weight_decay_on_bias_and_norm False \
    --exp_name tulu3.1_8b_grpo_fast_no_weight_decay \
    --image costah/open_instruct_dev0316_12 | uv run bash

# (not part of the benchmark) 2 nodes
python launch.py scripts/train/tulu3/grpo_fast_8b.sh \
    --cluster ai2/augusta-google-1 \
    --priority normal \
    --wandb_project_name open_instruct_public \
    --exp_name tulu3.1_8b_grpo_fast_axis_1_mean \
    --masked_mean_axis 1 \
    --image costah/open_instruct_dev0316_11 | uv run bash


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
