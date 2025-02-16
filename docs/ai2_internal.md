# Beaker job submission scripts

This document details some best practices when submitting jobs in our cluster.

## First-time setup

You need to first obtain API key or tokens from the following website:

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

## Job submission

`mason.py` is our job submission script. It takes in the command after `--` and runs it in the specified clusters. During the job submission, it automatically tries to setup a shared Hugging Face cache with environment variables. For example, it sets
* `HF_HOME=/weka/oe-adapt-default/allennlp/.cache/huggingface`. 
* `HF_DATASETS_CACHE=/weka/oe-adapt-default/allennlp/.cache/huggingface`
* `HF_HUB_CACHE=/weka/oe-adapt-default/allennlp/.cache/hub`


You can run things like below for a quick spin. This example just starts a beaker job to print the Python version in your beaker image.

```bash
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 ai2/saturn-cirrascale ai2/neptune-cirrascale \
    --workspace ai2/tulu-3-dev \
    --image nathanl/open_instruct_auto --pure_docker_mode \
    --priority normal \
    --budget ai2/oe-adapt \
    --gpus 1 -- which python
```


### Caching model (Weka-only)

You can run the following command to cache models to the share Hugging Face cache. We recommend this for weka (shared filesystem) users because otherwise your jobs would 1) waste time downloading the model while GPU is idle and 2) risk potential download failures.

```bash
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 ai2/saturn-cirrascale ai2/neptune-cirrascale --image nathanl/open_instruct_auto --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority normal \
    --preemptible \
    --budget ai2/allennlp \
    --gpus 0 -- python scripts/cache_hf.py \
    --model_name_or_path "allenai/open_instruct_dev" \
    --model_revision "reward_modeling__1__1737836233"
```

If you have the weka environment setup you can also just run

```bash
python scripts/cache_hf.py \
    --model_name_or_path "allenai/open_instruct_dev" \
    --model_revision "reward_modeling__1__1737836233"
```


### Supervised Fine-tuning (SFT):

Otherwise, the `mason.py` command can be used to launch all of our other training jobs.
```bash
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority normal \
    --image nathanl/open_instruct_auto --pure_docker_mode \
    --preemptible \
    --num_nodes 4 \
    --budget ai2/oe-adapt \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --tokenizer_name meta-llama/Llama-3.1-8B \
    --use_slow_tokenizer \
    --use_flash_attn \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-06 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --output_dir output/sft_8b \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --reduce_loss sum \
    --model_revision main \
    --dataset_mixer_list allenai/tulu-3-sft-mixture 1.0 \
    --dataset_mix_dir output/sft_8b \
    --exp_name tulu-3-8b-sft \
    --seed 123
```


Note that during job submission, we will try to tokenize and cache the dataset so we are not running these CPU-heavy workloads in GPU jobs. Specifically, `mason.py` will parse out `python` command you are running and attempts to run it with `--cache_dataset_only` flag. For example, you will see output like

```bash
📦📦📦 Running the caching full_command: python open_instruct/dpo_tune_cache.py --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT --use_flash_attn --tokenizer_name allenai/Llama-3.1-Tulu-3-8B-SFT --max_seq_length 2048 --preprocessing_num_workers 16 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 5e-07 --lr_scheduler_type linear --warmup_ratio 0.1 --weight_decay 0.0 --num_train_epochs 1 --output_dir output/dpo_8b --with_tracking --report_to wandb --logging_steps 1 --model_revision main --gradient_checkpointing --dataset_mixer_list allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0 --use_slow_tokenizer --use_lora False --dpo_loss_type dpo_norm --dpo_beta 5 --exp_name tulu-3-8b-dpo --cache_dataset_only
[2025-01-21 09:58:09,342] [WARNING] [real_accelerator.py:162:get_accelerator] Setting accelerator to CPU. If you have GPU or other accelerator, we were unable to detect it.
[2025-01-21 09:58:09,354] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cpu (auto detect)
Failed to get Beaker experiment: b'Error: experiment "01JD3WCQYTBPE195GVWPVMDHVV" not found\n\n'

# ....
Cache not found, transforming datasets...
Map (num_proc=192): 100%|██████████████████| 271409/271409 [00:40<00:00, 6690.00 examples/s]
Filter (num_proc=192): 100%|███████████████| 271409/271409 [00:36<00:00, 7492.34 examples/s]
Creating parquet from Arrow format: 100%|███████████████████| 17/17 [00:02<00:00,  7.26ba/s]
Creating parquet from Arrow format: 100%|███████████████████| 17/17 [00:02<00:00,  7.77ba/s
🚀 Pushed transformed dataset to https://huggingface.co/datasets/vwxyzjn/dataset-mix-cached/tree/992c2b51ba
✅ Found cached dataset at https://huggingface.co/datasets/vwxyzjn/dataset-mix-cached/tree/992c2b51ba

# ...

Kicked off Beaker job. https://beaker.org/ex/01JJ50D88M757GZD14W9CNN7NT
```

It would be most helpful if you run the `mason.py` command on a vscode session with access to weka, that way, the dataset is also automatically downloaded to the shared `HF_HOME` on weka, etc.

When you inspect the job, it's going to have the following outputs, meaning the cached dataset is found and used:

```
2025-01-21T18:02:04.840723691Z 
2025-01-21T18:02:05.948433221Z ✅ Found cached dataset at https://huggingface.co/datasets/vwxyzjn/dataset-mix-cached/tree/992c2b51ba
2025-01-21T18:02:06.120806993Z ✅ Found cached dataset at https://huggingface.co/datasets/vwxyzjn/dataset-mix-cached/tree/992c2b51ba
2025-01-21T18:02:06.190569046Z ✅ Found cached dataset at https://huggingface.co/datasets/vwxyzjn/dataset-mix-cached/tree/992c2b51ba
2025-01-21T18:02:06.197208582Z ✅ Found cached dataset at https://huggingface.co/datasets/vwxyzjn/dataset-mix-cached/tree/992c2b51ba
2025-01-21T18:02:06.333301775Z ✅ Found cached dataset at https://huggingface.co/datasets/vwxyzjn/dataset-mix-cached/tree/992c2b51ba
2025-01-21T18:02:06.338503095Z ✅ Found cached dataset at https://huggingface.co/datasets/vwxyzjn/dataset-mix-cached/tree/992c2b51ba
2025-01-21T18:02:06.385010439Z ✅ Found cached dataset at https://huggingface.co/datasets/vwxyzjn/dataset-mix-cached/tree/992c2b51ba
```


### Direct Preference Optimization (DPO):


```bash
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority normal \
    --image nathanl/open_instruct_auto --pure_docker_mode \
    --preemptible \
    --num_nodes 4 \
    --budget ai2/oe-adapt \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \
    --use_flash_attn \
    --tokenizer_name allenai/Llama-3.1-Tulu-3-8B-SFT \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-07 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir /output \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --model_revision main \
    --gradient_checkpointing \
    --dataset_mixer_list allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0 \
    --use_slow_tokenizer \
    --use_lora False \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --exp_name tulu-3-8b-dpo
```


## RM:

```bash
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority high \
    --preemptible \
    --image nathanl/open_instruct_auto --pure_docker_mode \
    --budget ai2/oe-adapt \
    --num_nodes 4 \
    --gpus 8 -- accelerate launch \
    --config_file configs/ds_configs/deepspeed_zero3.yaml open_instruct/reward_modeling.py \
    --dataset_mixer '{"allenai/llama-3.1-tulu-3-8b-preference-mixture": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
    --dataset_eval_splits test_prefs \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \
    --chat_template tulu \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --num_train_epochs 1 \
    --output_dir output/rm_8b \
    --gradient_checkpointing \
    --push_to_hub \
    --with_tracking
```

## RLVR:

```bash
# make sure to match up the GPUs. E.g.,
# `--actor_num_gpus_per_node 6 8`
# `--vllm_tensor_parallel_size 2`
# translates to 6 + 2 + 8 = 16 GPUs
# which matches up with `--num_nodes 2 --gpus 8`
for beta in 0.05; do
exp_name="0112_ppo_rlvr_${beta}_${RANDOM}"
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority high \
    --preemptible \
    --image nathanl/open_instruct_auto --pure_docker_mode \
    --budget ai2/oe-adapt \
    --num_nodes 2 \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --exp_name $exp_name \
    --beta $beta \
    --output_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint/$exp_name \
    --try_launch_beaker_eval_jobs_on_weka \
    --try_launch_beaker_eval_jobs False \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-DPO \
    --reward_model_path allenai/Llama-3.1-Tulu-3-8B-RM \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template_name tulu \
    --sft_messages_key messages \
    --learning_rate 3e-7 \
    --total_episodes 200000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 1 \
    --local_rollout_forward_batch_size 1 \
    --local_mini_batch_size 4 \
    --local_rollout_batch_size 4 \
    --actor_num_gpus_per_node 6 8 \
    --vllm_tensor_parallel_size 2 \
    --vllm_enforce_eager \
    --apply_verifiable_reward true \
    --seed 3 \
    --num_evals 1000 \
    --save_freq 40 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
done
```

GRPO:
```bash
for beta in 0.0 0.05 0.03; do
for nspp in 4 8 16; do
exp_name="0112_grpo_math_zs_${beta}_${nspp}_${RANDOM}"
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority high \
    --preemptible \
    --num_nodes 4 \
    --budget ai2/oe-adapt \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& uv run python open_instruct/grpo_vllm_thread_ray_gtrl.py \
    --exp_name $exp_name \
    --beta $beta \
    --local_mini_batch_size 8 \
    --number_samples_per_prompt $nspp \
    --output_dir /weka/oe-adapt-default/costah/$exp_name \
    --dataset_mixer "{\"ai2-adapt-dev/math_ground_truth_zs\": 1.0}" \
    --dataset_train_splits train \
    --dataset_eval_mixer "{\"ai2-adapt-dev/math_ground_truth_zs\": 32}" \
    --dataset_eval_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 4096 \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-DPO \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template_name tulu \
    --sft_messages_key messages \
    --learning_rate 5e-7 \
    --total_episodes 1000000 \
    --penalty_reward_value 0.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_rollout_batch_size 8 \
    --actor_num_gpus_per_node 7 8 8 8 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --num_evals 1000 \
    --save_freq 40 \
    --reward_model_multiplier 0.0 \
    --no_try_launch_beaker_eval_jobs \
    --try_launch_beaker_eval_jobs_on_weka \
    --gradient_checkpointing \
    --with_tracking
done
done
```


### End-to-end Model Training

For post-training, we often need to train the models throughout all 3 stages. The rough steps are as follows:

1. Run a sweep of SFT training and use the internal leaderboard https://huggingface.co/spaces/allenai/oe-eval-leaderboard to select the best model.
2. Run a sweep of DPO training and select the best model.
3. Based on the best DPO model, use its dataset to train an RM.
4. Use the best DPO (and RM) to train an RLVR model.


We have some example dev scripts on the whole process in the `docs/archived_dev_scripts` directory. Note that these scripts are not cleaned up like [docs/tulu3.md](docs/tulu3.md), but they are useful for reference.

* docs/archived_dev_scripts/olmo2_1124.sh (the commands used to produce [OLMo 2 1124](https://huggingface.co/collections/allenai/olmo-2-674117b93ab84e98afc72edc))
* docs/archived_dev_scripts/olmoe_0125.sh (the commands used to produce [OLMoE 0125](https://huggingface.co/collections/allenai/olmoe-0125-67992134f9ebea0a941706ca))


### Ai2 Internal Evaluation

We provide a script integrated with beaker for use internally at Ai2. For example, to run all the tulu 3 evals with easy uploading:
```bash
python scripts/submit_eval_jobs.py \
      --model_name <model name> \
      --location <beaker id> \
      --is_tuned --workspace tulu-3-results \
      --preemptible \
      --use_hf_tokenizer_template \
      --beaker_image nathanl/open_instruct_auto \
      --upload_to_hf allenai/tulu-3-evals \
      --run_oe_eval_experiments \
      --run_safety_evaluations \
      --skip_oi_evals
```
Replace location with your beaker ID, and model name with your model name (note this will affect experiment naming, so make it unique and memorable!). For HF models, use a name with `hf-<model-name>` for the model_name argument, and for location give the HF path (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`). Note this assumes your model has a valid HF tokenizer chat template.

To make this script work you have to clone the [following repository](https://github.com/allenai/oe-eval-internal/tree/main) to the top level directory of the open-instruct repository.

You can additionally run other evaluations in this repository through varied arguments to the script.

You can also upload metadata via the `scripts/add_metadata.py` script. Just run `python scripts/add_metadata.py` and follow the prompts.

If you have used automatic evaluation, you cacn also upload metadata via `python add_metadata_from_wandb.py`. Example usage:

```bash
# from a wandb url
python scripts/add_metadata_from_wandb.py --wandb_run_id ai2-llm/open_instruct_internal/runs/fjclmg47
# or from a hf_revision (the name of the autoeval)
python scripts/add_metadata_from_wandb.py --hf_repo_revision valpy_dpo_mix_uf_wc_regen_da_sftmix_v4.23___model__42__1725581304
```

## Running with gantry

You can also run with gantry, if you want to test changes.
**Important**: Before you run any command with gantry, make sure you *commit and push*, since gantry will attempt to clone the repo with your local latest commit hash.

See the "One-Time Setup" section below before running commands. To test your setup, run the following command -- if this job succeeds, then you're ready to run evaluations with gantry.

```bash
gantry run --workspace {workspace} --budget ai2/oe-adapt --beaker-image kavelr/oe-safety --venv base --cluster ai2/mosaic-cirrascale --env-secret OPENAI_API_KEY=openai_api_key --env-secret HF_TOKEN=hf_token -- python -c 'print("Hello world")'
```

You can freely add any additional arguments to give to Beaker, such as a `--priority` tag which can be set to preemptible, normal, high, or urgent. AI2 policies may restrict the priorities that are available to users on certain clusters.

In the examples below, text within {} tags should be replaced with your own values. 

As a convenience, you can use the `evaluation/gantry_run.sh` script which includes some necessary arguments. You can use it the same way as `gantry run`, but excluding these boilerplate arguments (take a look at the script to see what it includes). Example usage:

```bash
PYTHONPATH=safety-eval ./evaluation/gantry_run.sh --workspace {workspace} --cluster {cluster} --gpus {n_gpus} \
    --priority {priority} -- python evaluation/run_all_generation_benchmarks.py \
    --model_name_or_path allenai/tulu-2-dpo-7b \
    --model_input_template_path_or_name tulu2 \
    --report_output_path /results/metrics.json
```

### Extra Beaker Commands
Here is an example using the full `gantry run` command. Use the beaker image `seungjuh/oe-safety-support-olmo17`

**Important**: Please include all the beaker arguments exactly as in the examples unless intentionally modifying some configuration. Many of them are necessary to avoid job failures, such as `--beaker-image`, `--venv`, and `--env-secret`. Note that `openai_api_key` and `hf_token` are Beaker workspace secret names, so should *not* be replaced with actual values (see One-Time Setup).

Note that the `--` divides the gantry command from the evaluation command - you can edit the second part to run whatever eval suite you want from the `eval.py` script. Any additional Beaker arguments such as a dataset mount to use a model from a Beaker dataset or adding a priority tag can be added before the `--`.

You can also run all generator evaluations parallelized across the GPUs allocated to your batch job, like so:
```bash
gantry run --workspace {your_workspace} --cluster {cluster} --gpus {n_gpus} \
    --name {beaker_experiment_name} --task-name {beaker_task_name} --beaker-image seungjuh/oe-safety-support-olmo17 --venv base \
    --env-secret OPENAI_API_KEY=openai_api_key \
    --env-secret HF_TOKEN=hf_token \
    --budget {budget} -- python evaluation/run_all_generation_benchmarks.py \
    --model_name_or_path allenai/tulu-2-dpo-7b \
    --model_input_template_path_or_name tulu2 \
    --report_output_path /results/metrics.json --save_individual_results_path /results/all.json
```

Because the `--report_output_path` argument is set to `/results/metrics.json`, the output will automatically get logged to Beaker metrics in the experiment page ([example](https://beaker.org/ex/01HW8NKZ458MA1PSB1X4YQTH94/tasks/01HW8NKZ4DTDA8FEFDGWA7Q8XX/job/01HW8NM2QR5AYB53PYP32J2VAA)).


### Common Gotchas

If you're experiencing job failures, here are some things to check:

- Make sure your local changes are committed,  pushed, and up to date with the remote
- Make sure you have `--beaker-image seungjuh/oe-safety-support-olmo17` and `--venv base` in your `gantry run` command
- Check your GitHub personal access token is authorized to access the allenai organization
- Make sure the openai_api_key and hf_token secrets exist in your Beaker workspace