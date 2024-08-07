# Scripts Docs

There are many scripts in this repo, serving many different purposes. Here's a breakdown of the most important training scripts and how to use them. Generally, they are split into the following categories:
1. Instruction training.
2. Direct Preference Optimization (DPO) training.
3. Submitting jobs on Ai2 infrastructure (Beaker). **Use this type of script for launching multiple jobs easily)
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
4. `finetune_with_acceralate.sh`: Script that the `_config` option above is based on. Uses options provided at CLI. **Change hyperparameters by manually editing or copying the script**.

## Direct Preference Optimization (DPO) scripts
1. `dpo_train_with_accelerate_config.sh`: Script for running `open_instruct/dpo_tune.py` with configs found in `configs/train_configs/dpo/`. Good for reproducing results. E.g.
```bash
sh scripts/dpo_train_with_accelerate_config.sh 8 configs/train_configs/dpo/default.yaml
```
2. `dpo_train_with_accelerate.sh`: Script for running `open_instruct/dpo_tune.py` directly. **Change hyperparameters by manually editing or copying the script**.
E.g.
```bash
sh scripts/dpo_train_with_accelerate.sh
```
3. `dpo_train_with_qlora.sh`: Same as (2) with QLoRA quantization.

## Beaker / job submission scripts
1. `submit_eval_jobs.py`: Submit eval jobs for tasks in `scripts/evals/`. TODO add example.
2. `submit_finetune_jobs.py`: **Core script** for submitting multiple and configurable instruction tuning jobs. This script works for both single- and multi-node configurations. It by default reads configs in `configs/train_configs`, but also can take in CLI arguments matching those in `open_instruct/utils.py` `FlatArguments` class. 
Example of running this is in `scripts/submit_finetune_jobs.sh`. 
```
python scripts/submit_finetune_job.py --config=configs/train_configs/sft/default.yaml  --learning_rate 1e-6
python scripts/submit_finetune_job.py --config=configs/train_configs/sft/default.yaml  --learning_rate 4e-6
python scripts/submit_finetune_job.py --config=configs/train_configs/sft/default.yaml  --learning_rate 1e-5
python scripts/submit_finetune_job.py --config=configs/train_configs/sft/default.yaml  --learning_rate 4e-5
```

## Other
1. `collect_eval_results.py`: For collating metrics from `open-instruct` evaluation job. E.g.
```bash
python scripts/collect_eval_results.py \
    --experiment_id 01HV0P4E3MW9211HX0JEKM0PXM \
    --job_suffix _tulu2_13b_dpo_ultrainteract_04082024 \
    --output_file metrics.json \
    --task_order gsm_cot gsm_direct toxigen alpaca_eval \
    --print_table \
    --table_file metrics.tsv
```
2. `weights/weight_diff.py`: For converting weight diffs (as used with LLaMA 1) to full weights for eval/use. E.g.
```bash
python scripts/weights/weight_diff.py recover --path_raw ${hf_llama_path} --path_tuned ${output_path} --path_diff ${diff_location}
```
3. `weights/convert_llama_weights_to_hf.sh`: Use `transformers` to convert weights.
4. `data/*`: scripts for inpecting statistics of and rebuilding Tulu 1/2/N datasets from scratch (where possible).
