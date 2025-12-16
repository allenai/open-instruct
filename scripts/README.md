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

```bash
sh scripts/finetune_with_accelerate_config.sh 1 configs/train_configs/sft/mini.yaml
sh scripts/finetune_with_accelerate_config.sh 1 configs/train_configs/sft/default.yaml
sh scripts/finetune_with_accelerate_config.sh 8 configs/train_configs/sft/olmo_17_sft.yaml
```

4. `finetune_with_acceralate.sh`: Script that the `_config` option above is based on. Uses options provided at CLI. **Change hyperparameters by manually editing or copying the script**.

## Direct Preference Optimization (DPO) scripts

1. `dpo_train_with_accelerate_config.sh`: Script for running `open_instruct/dpo_tune.py` with configs found in `configs/train_configs/dpo/`. Good for reproducing results. E.g.
```bash
sh scripts/dpo_train_with_accelerate_config.sh 1 configs/train_configs/dpo/mini.yaml
sh scripts/dpo_train_with_accelerate_config.sh 1 configs/train_configs/dpo/default.yaml
sh scripts/dpo_train_with_accelerate_config.sh 8 configs/train_configs/dpo/default.yaml
```

2. `dpo_train_with_accelerate.sh`: Script for running `open_instruct/dpo_tune.py` directly. **Change hyperparameters by manually editing or copying the script**.
E.g.
```bash
sh scripts/dpo_train_with_accelerate.sh
```
3. `dpo_train_with_qlora.sh`: Same as (2) with QLoRA quantization.

## Beaker / job submission scripts


0. First-time setup: You need to first obtain API key or tokens from the following website:

* `BEAKER_TOKEN`: https://beaker.org/user
* `WANDB_API_KEY`: https://wandb.ai/authorize
* `HF_TOKEN`: https://huggingface.co/settings/tokens

Then you need to write them in beaker secret as follows (replace the `xxxx` with your own API key or token)
```bash
beaker_whoami=$(beaker account whoami --format json | jq -r '.[0].name')
beaker secret write -w ai2/tulu-2-improvements "${beaker_whoami}_BEAKER_TOKEN" xxxx
beaker secret write -w ai2/tulu-2-improvements "${beaker_whoami}_WANDB_API_KEY" xxxx
beaker secret write -w ai2/tulu-2-improvements "${beaker_whoami}_HF_TOKEN" xxxx
```


1. `submit_eval_jobs.py`: Submit eval jobs for tasks in `scripts/evals/`. For example, llama 3 tulu 2 and upload to the tulu-3 eval database.
```bash
# submit evals on a model in beaker dataset
python scripts/submit_eval_jobs.py --model_name llama_31_tulu_2_8b --location 01J4MGRSS3FM1J4E6XSH3459DK --is_tuned --workspace tulu-3-results --preemptible --use_hf_tokenizer_template --beaker_image nathanl/open_instruct_auto --upload_to_hf allenai/tulu-3-evals

# submit evals on a model in huggingface; note you need to 1) prepend the model name with `hf-` and 2) replace `--location` with the hf repo id
python scripts/submit_eval_jobs.py --model_name hf-llama_31_tulu_2_8b --location allenai/llama-3-tulu-2-8b --is_tuned --workspace tulu-3-results --preemptible --use_hf_tokenizer_template --beaker_image nathanl/open_instruct_auto --upload_to_hf allenai/tulu-3-evals
python scripts/submit_eval_jobs.py --model_name hf-llama_31_tulu_2_8b --location vwxyzjn/online_dpo_tulu_2 --is_tuned --workspace tulu-3-results --preemptible --use_hf_tokenizer_template --beaker_image nathanl/open_instruct_auto --upload_to_hf allenai/tulu-3-evals

python scripts/submit_eval_jobs.py --model_name hf-online-dpo-llama-tulu2-longer --beaker_image costah/open_instruct_test --location vwxyzjn/online_dpo_vllm__allenai_llama-3-tulu-2-8b --hf_revision online_dpo_vllm__1__1724038538 --is_tuned --workspace tulu-3-results --preemptible --use_hf_tokenizer_template --upload_to_hf allenai/tulu-3-evals

```
Here, it is important to know that for using `oe-eval`, normally we run `--skip_oi_evals`, `run_safety_evaluations`, and `run_oe_eval_experiments`.

2. `submit_finetune_jobs.py`: **Core script** for submitting multiple and configurable instruction tuning jobs. This script works for both single- and multi-node configurations. It by default reads configs in `configs/train_configs`, but also can take in CLI arguments matching those in `open_instruct/utils.py` `FlatArguments` class.
Example of running this is in `scripts/submit_finetune_jobs.sh`.
```
python scripts/submit_finetune_job.py --config=configs/train_configs/sft/default.yaml  --learning_rate 1e-6 --exp_name sft_lr_search
python scripts/submit_finetune_job.py --config=configs/train_configs/sft/default.yaml  --learning_rate 4e-6 --exp_name sft_lr_search
python scripts/submit_finetune_job.py --config=configs/train_configs/sft/default.yaml  --learning_rate 1e-5 --exp_name sft_lr_search
python scripts/submit_finetune_job.py --config=configs/train_configs/sft/default.yaml  --learning_rate 4e-5 --exp_name sft_lr_search
```


You may want to add the `--exp_name`, the name that appears in the internal leaderboard.

<img width="1132" alt="image" src="https://github.com/user-attachments/assets/f99ff0d6-5436-4932-9fa7-d6266a68fba0">

<img width="1294" alt="image" src="https://github.com/user-attachments/assets/17251833-e90f-44d1-88f9-dd12c9465914">




To use this for multi-node jobs, here is an example that runs IFT on 4 nodes:
```
python scripts/submit_finetune_job.py --default_beaker_config configs/beaker_configs/default_finetune_multinode.yaml --config configs/train_configs/sft/tulu3_8b_preview_mix_v3.1.yaml --cluster ai2/jupiter --workspace ai2/tulu-3-dev --num_nodes 4 --exp_name preview_mix
```

3. `submit_dpo_job.py`: **Core script** for submitting DPO tuning jobs. It should behave like the finetune script, but additionally can take in beaker datasets to mount via `--datasets`, e.g.:
```
python scripts/submit_dpo_job.py --config configs/train_configs/dpo/my_dpo_config.yaml --datasets my_beaker_id:/model --experiment_name my_experiment_name
```
In this case, we also ask you provide an experiment name, as we don't know the name of the model being finetuned if it is mounted to `/model`.


### Docker-less job submssions

It is possible to re-use the existing environment you have and run things without having to build a docker container. The idea is to install python on NFS. You can refer to https://gist.github.com/vwxyzjn/58a2714cf3fbab5bf672ff750e86a537 for more detail.

Then you can submit jobs via `mason.py`, which we modified from https://github.com/allenai/mason. You can run the following to do a quick check
```bash
python mason.py \
    --cluster ai2/jupiter \
    --priority low \
    --budget ai2/jupiter \
    --gpus 1 -- which python
```

If you are successful in setting up python on NFS, your `which python` should match the `which python` output in the beaker job.

![image](https://github.com/user-attachments/assets/4f37d5bd-64bd-476b-9dad-1e35795b2618)



After setting it up successfully, say you are running `sh scripts/dpo_train_with_accelerate_config.sh 8 configs/train_configs/dpo/default.yaml` locally, now you can submit batch jobs via

```bash
python mason.py \
    --cluster ai2/jupiter \
    --priority low \
    --budget ai2/jupiter \
    --gpus 1 -- sh scripts/finetune_with_accelerate_config.sh 1 configs/train_configs/sft/mini.yaml
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

## Notes on data mixing
Most of the scripts with `_config` take in configs that look like the following (just the data part):
```
dataset_mixer:
 allenai/tulu-v2-sft-mixture: 0.5
 HuggingFaceH4/no_robots: 0.8
```
There are many ways to configure data mixing. This is done with fractions, but also they can be done with number of samples directly. E.g.
```
dataset_mixer:
 allenai/tulu-v2-sft-mixture: 50000
 HuggingFaceH4/no_robots: 2500
```
The mixer is the advanced alternate to existing data arguments (which are still compatible, for reproducibility), such as local files:
```
train_file: data/processed/tulu_v2/tulu_v2_data.jsonl
```
or single HuggingFace datasets,
```
dataset_name: allenai/tulu-v2-sft-mixture
```
**Currently the dataset mixer is only supported for SFT models, but this will be expanded.**
With these options, the script will fail if multiple data args are passed, in the list of `dataset_mixer`, `train_file`, or `dataset_name`.
An internal arg, `dataset_mixer_list` was created to handle conversion from dict to string for Beaker jobs.
