# ------------------------------------------------------------
# Tulu3

# 8 nodes
python launch.py scripts/train/tulu3/finetune_8b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --seed 1 \
    --image costah/open_instruct_dev0320_11 | uv run bash

# 4 nodes
python launch.py scripts/train/tulu3/dpo_8b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --seed 1 \
    --image costah/open_instruct_dev0320_11 | uv run bash


# 2 nodes
python launch.py scripts/train/tulu3/reward_modeling_8b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority high \
    --seed 1 \
    --image costah/open_instruct_dev0320_11 | uv run bash

# 2 nodes
python launch.py scripts/train/tulu3/grpo_8b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --seed 40 \
    --image costah/open_instruct_dev0320_11 | uv run bash


# 2 nodes
python launch.py scripts/train/tulu3/ppo_8b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority high \
    --image costah/open_instruct_dev0320_13 | uv run bash

# 2 nodes
python launch.py scripts/train/tulu3/grpo_fast_8b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --seed 40 \
    --image costah/open_instruct_dev0320_11 | uv run bash

# 1 node
python launch.py scripts/train/tulu3/grpo_fast_8b_single_node.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --seed 40 \
    --image costah/open_instruct_dev0320_11 | uv run bash


# ------------------------------------------------------------
# Qwen

# 2 nodes
python launch.py scripts/train/qwen/grpo_fast_7b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0320_13 | uv run bash


# 2 nodes
python launch.py scripts/train/qwen/grpo_7b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0320_11 | uv run bash

# ------------------------------------------------------------
# Olmo2

# 8 nodes
python launch.py scripts/train/olmo2/finetune_13b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0320_11 | uv run bash

# 8 nodes
python launch.py scripts/train/olmo2/finetune_7b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0320_11 | uv run bash

# 8 nodes
python launch.py scripts/train/olmo2/finetune_32b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0320_11 | uv run bash

# 4 nodes
python launch.py scripts/train/olmo2/dpo_7b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0320_11 | uv run bash

# 4 nodes
python launch.py scripts/train/olmo2/dpo_13b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0320_11 | uv run bash

# 2 nodes
for lr in 3e-7 4e-7 5e-7; do 
    python launch.py scripts/train/olmo2/grpo_13b.sh \
        --cluster ai2/augusta-google-1 \
        --wandb_project_name open_instruct_public \
        --priority normal \
        --image costah/open_instruct_dev0320_11 costah/open_instruct_dev0316_18 \
        --learning_rate $lr | uv run bash
done
    
# 2 nodes
for lr in 3e-7 4e-7 5e-7; do 
    python launch.py scripts/train/olmo2/grpo_7b.sh \
        --cluster ai2/augusta-google-1 \
        --wandb_project_name open_instruct_public \
        --priority normal \
        --image costah/open_instruct_dev0320_11 costah/open_instruct_dev0316_18 \
        --learning_rate $lr | uv run bash
done






python launch.py scripts/train/tulu3/grpo_fast_8b.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --seed 3 \
    --beta 0.00 \
    --masked_mean_axis 1 \
    --image costah/open_instruct_dev0320_11 | uv run bash

# 2 nodes
python launch.py scripts/train/tulu3/grpo_fast_8b.sh \
    --cluster ai2/augusta-google-1 \
    --priority normal \
    --wandb_project_name open_instruct_public \
    --exp_name tulu3.1_8b_grpo_fast_axis_1_mean \
    --masked_mean_axis 1 \
    --image costah/open_instruct_dev0320_11 | uv run bash


# ------------------------------------------------------------
# single gpu experiments
python launch.py scripts/train/qwen/grpo_fast_3b_single_node.sh \
    --cluster ai2/augusta-google-1 \
    --wandb_project_name open_instruct_public \
    --priority normal \
    --image costah/open_instruct_dev0320_11 | uv run bash





python launch.py scripts/train/olmo2/grpo_fast_32b_tulu.sh \
    --cluster ai2/augusta-google-1 \
    --priority urgent \
    --image costah/open_instruct_dev0320_11 | uv run bash




for seed in 1 2 3; do 
    python launch.py scripts/train/qwen/grpo_fast_7b_orz.sh \
        --cluster ai2/augusta-google-1 \
        --exp_name qwen2.5_7b_grpo_fast_orz_seed_$seed \
        --priority normal \
        --image costah/open_instruct_dev0320_12  \
        --seed $seed | uv run bash
done

for seed in 1 2 3; do 
    python launch.py scripts/train/qwen/grpo_fast_7b_orz.sh \
        --cluster ai2/augusta-google-1 \
        --exp_name qwen2.5_7b_grpo_fast_mm1_orz_seed_$seed \
        --masked_mean_axis 1 \
        --priority normal \
        --image costah/open_instruct_dev0320_12  \
        --seed $seed | uv run bash
done

for seed in 1 2 3; do 
    python launch.py scripts/train/qwen/grpo_fast_7b_orz.sh \
        --cluster ai2/augusta-google-1 \
        --exp_name qwen2.5_7b_drgrpo_fast_orz_seed_$seed \
        --dr_grpo True \
        --priority normal \
        --image costah/open_instruct_dev0320_12  \
        --seed $seed | uv run bash
done

for seed in 1 2 3; do 
    python launch.py scripts/train/qwen/grpo_fast_7b_orz.sh \
        --cluster ai2/augusta-google-1 \
        --exp_name qwen2.5_7b_drgrpo_fast_mm1_orz_seed_$seed \
        --dr_grpo True \
        --masked_mean_axis 1 \
        --priority normal \
        --image costah/open_instruct_dev0320_12  \
        --seed $seed | uv run bash
done


for response_length in 4096 8192; do 
    python launch.py scripts/train/qwen/grpo_fast_7b_orz.sh \
        --cluster ai2/augusta-google-1 \
        --exp_name qwen2.5_7b_drgrpo_fast_mm1_orz_response_length_$response_length \
        --dr_grpo True \
        --masked_mean_axis 1 \
        --response_length $response_length \
        --priority normal \
        --image costah/open_instruct_dev0320_12 | uv run bash
done


for seed in 1 2 3; do 
    python launch.py scripts/train/qwen/grpo_fast_7b_orz.sh \
        --cluster ai2/augusta-google-1 \
        --exp_name qwen2.5_7b_drgrpo_correct_fast_mm1_orz_seed_$seed \
        --dr_grpo True \
        --masked_mean_axis 1 \
        --priority normal \
        --image costah/open_instruct_dev0320_13  \
        --seed $seed | uv run bash
done




for seed in 1 2 3; do 
    python launch.py scripts/train/qwen/grpo_fast_7b_orz.sh \
        --cluster ai2/augusta-google-1 \
        --exp_name 0324_qwen2.5_7b_grpo_small_scalar_seed_$seed \
        --masked_mean_axis 1 \
        --priority normal \
        --verification_reward 1.0 \
        --apply_r1_style_format_reward False \
        --image costah/open_instruct_dev0320_13  \
        --seed $seed | uv run bash
done
