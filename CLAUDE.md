# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Open Instruct is a comprehensive framework for instruction-tuning and post-training language models, developed by AllenAI. It includes implementations for:
- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)
- Reinforcement Learning with Verifiable Rewards (RLVR)
- PPO and GRPO implementations
- Comprehensive evaluation suite

## Development Commands

### Code Quality
```bash
# Format code
make style

# Run quality checks
make quality

# Run tests
uv run pytest
```

### Training Commands

#### Quick Debug Runs (1 GPU)
```bash
# Supervised Fine-Tuning
accelerate launch --mixed_precision bf16 --num_processes 1 open_instruct/finetune.py \
    --model_name_or_path EleutherAI/pythia-14m \
    --dataset_mixer_list allenai/tulu-3-sft-personas-algebra 100 \
    --output_dir output/debug_sft/

# DPO
accelerate launch --mixed_precision bf16 --num_processes 1 open_instruct/dpo_tune.py \
    --model_name_or_path EleutherAI/pythia-14m \
    --dataset_mixer_list allenai/tulu-3-ultrafeedback-cleaned-binarized 100 \
    --output_dir output/debug_dpo/

# GRPO/RLVR (main RL method)
python open_instruct/grpo_fast.py \
    --model_name_or_path HuggingFaceTB/SmolLM2-135M \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --total_episodes 200 \
    --num_evals 20 \
    --single_gpu_mode \
    --deepspeed_stage 2 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.5

# PPO (alternative RL method)
accelerate launch --num_processes 1 open_instruct/ppo_vllm_thread.py \
    --model_name_or_path EleutherAI/pythia-14m \
    --reward_model_path EleutherAI/pythia-14m \
    --dataset_mixer_list allenai/tulu-3-sft-personas-algebra 100 \
    --output_dir output/debug_ppo/
```

#### Production Training (8 GPUs)
```bash
# See scripts/train/tulu3/*.sh for full examples
bash scripts/train/tulu3/finetune_8b.sh  # SFT
bash scripts/train/tulu3/dpo_8b.sh        # DPO
bash scripts/train/tulu3/grpo_fast_8b.sh  # GRPO/RLVR
```

### Evaluation Commands
```bash
# MMLU evaluation
python -m eval.mmlu.run_eval \
    --model_name_or_path <model_path> \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/<model_name>

# GSM8K evaluation
python -m eval.gsm.run_eval \
    --model_name_or_path <model_path> \
    --data_dir data/eval/gsm \
    --save_dir results/gsm/<model_name>

# HumanEval evaluation
python -m eval.codex_humaneval.run_eval \
    --model_name_or_path <model_path> \
    --save_dir results/humaneval/<model_name>
```

## Key Architecture Components

### Core Training Scripts
- `open_instruct/finetune.py`: Supervised fine-tuning implementation
- `open_instruct/dpo_tune.py`: Direct Preference Optimization
- `open_instruct/ppo_vllm_thread.py`: PPO with vLLM for efficient inference
- `open_instruct/grpo_fast.py`: Main GRPO/RLVR implementation with async rollouts
- `open_instruct/reward_modeling.py`: Reward model training

### Dataset Handling
- Uses `dataset_mixer_list` format: `"dataset_name ratio"` (e.g., `"allenai/tulu-3-sft-mixture 100"`)
- Supports multiple datasets with different sampling ratios
- Custom dataset processors in `open_instruct/dataset_processor.py`

### Model Support
- Primary models: Llama 3.1, OLMo 2, Qwen
- Flash Attention 2 support for efficiency
- LoRA/QLoRA support for parameter-efficient training
- DeepSpeed integration for large-scale training

### Distributed Training
- Uses HuggingFace Accelerate for distributed training
- DeepSpeed configs in `configs/ds_configs/`
- Multi-node training supported via Beaker (AI2 internal) or standard SLURM

## GRPO/RLVR Training (`grpo_fast.py`)

The main RL training script is `open_instruct/grpo_fast.py`, which implements Group Relative Policy Optimization with support for:

### Key Features
- **Async rollouts**: Can generate responses ahead of training for efficiency
- **Verifiable rewards**: Supports code execution, LLM judges, and format checking
- **Multi-GPU inference**: Uses vLLM for efficient distributed generation
- **Tool use**: Supports search and code tools during generation

### Important Parameters
```bash
# Core algorithm settings
--beta 0.01                    # KL penalty coefficient
--kl_estimator kl3             # KL estimation method (kl1-4)
--alpha 0.6                    # Polyak averaging for reference policy updates

# Generation settings  
--num_unique_prompts_rollout 48    # Unique prompts per rollout
--num_samples_per_prompt_rollout 16 # Samples per prompt (for best-of-n)
--response_length 2048             # Max response tokens
--temperature 1.0                  # Sampling temperature

# Distributed settings
--num_learners_per_node 6      # GPUs for training per node
--vllm_num_engines 10          # Number of vLLM engines
--vllm_tensor_parallel_size 1  # Tensor parallel size per engine
--deepspeed_stage 2            # DeepSpeed ZeRO stage

# Reward settings
--apply_verifiable_reward true # Enable code/math verification
--verification_reward 10.0     # Reward value for correct solutions
--non_stop_penalty true        # Penalize incomplete generations
```

### Debug vs Production
- **Debug**: Use `--single_gpu_mode` to run vLLM and training on same GPU
- **Production**: Use multiple vLLM engines across nodes for scale

## Important Notes

1. **Environment Setup**: The codebase requires Python 3.10+ and CUDA 12.1+. Use `uv sync` for dependency management or follow manual installation in README.

2. **Flash Attention**: Most training scripts expect Flash Attention 2. Install with:
   ```bash
   pip install flash-attn==2.7.2.post1 --no-build-isolation
   ```

3. **Evaluation**: For comprehensive evaluation, use [OLMES](https://github.com/allenai/olmes) instead of the built-in evaluation suite.

4. **Model Checkpoints**: Released models follow naming convention `allenai/{Model}-Tulu-3-{Size}-{Stage}` where Stage is SFT, DPO, or final (RLVR).

5. **Configuration**: Training configs use YAML format in `configs/train_configs/`. Most scripts accept command-line overrides for all parameters.

6. **Logging**: Supports Weights & Biases logging. Set `--report_to wandb` and ensure `WANDB_API_KEY` is set.

7. **Common Issues**:
   - OOM errors: Reduce `per_device_train_batch_size` or enable gradient checkpointing
   - Flash Attention issues: Ensure compatible GPU (Ampere or newer) or disable with `--use_flash_attn False`
   - Dataset loading: Check dataset names match HuggingFace Hub exactly