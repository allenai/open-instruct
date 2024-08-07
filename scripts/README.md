# Scripts Docs

There are many scripts in this repo, serving many different purposes. Here's a breakdown of the most important training scripts and how to use them. Generally, they are split into the following categories:
1. Instruction training.
2. Direct Preference Optimization (DPO) training.
3. Submitting jobs on Ai2 infrastructure (Beaker).
4. Data and results management. 

This readme covers each category and normal use-cases.

## Instruct training scripts
The following scripts are used for fine-tuning. 
For Ai2 users, these scripts all work best in interactive sessions (not in batch jobs). 

1. `finetune_lora_with_acceralate.sh`: Script for running `open_instruct/finetune.py` with LoRA.
2. `finetune_qlora_with_acceralate.sh`: Script for running `open_instruct/finetune.py` with QLoRA.
3. `finetune_with_acceralate_config.sh`: Script for running `open_instruct/finetune.py` with configs found in `configs/train_configs/sft/`. Good for reproducing results. Example usages:

```
sh scripts/finetune_with_accelerate_config.sh 1 configs/train_configs/sft/default.yaml
sh scripts/finetune_with_accelerate_config.sh 8 configs/train_configs/sft/olmo_17_sft.yaml
```
4. `finetune_with_acceralate.sh`: Script that the `_config` option above is based on. Uses options provided at CLI.
