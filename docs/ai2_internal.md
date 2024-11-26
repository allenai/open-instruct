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

### Gantry One-Time Setup

Before you can use gantry, there are a couple of things to set up. For the workspace you use, ensure it is owned by the `ai2` organization, or gantry won't be able to create the experiments.

1. Run `pip install beaker-gantry beaker-py`
2. Create a [GitHub personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) with "repo" scope
3. Go to https://github.com/settings/tokens and authorize your token to configure SSO access to the allenai organization
4. Run `gantry config set-gh-token` and paste the token created above when prompted
5. Create a [HuggingFace access token](https://huggingface.co/settings/tokens) with "read" scope (this is used to authenticate for using restricted models like Llama series)
6. Run `beaker secret write --workspace {your_workspace} hf_token {your_token}`
7. Obtain an OpenAI API key and run `beaker secret write --workspace {your_workspace} openai_api_key {your_api_key}

Doing these steps once will set up your workspace to use gantry.


### Common Gotchas

If you're experiencing job failures, here are some things to check:

- Make sure your local changes are committed,  pushed, and up to date with the remote
- Make sure you have `--beaker-image seungjuh/oe-safety-support-olmo17` and `--venv base` in your `gantry run` command
- Check your GitHub personal access token is authorized to access the allenai organization
- Make sure the openai_api_key and hf_token secrets exist in your Beaker workspace