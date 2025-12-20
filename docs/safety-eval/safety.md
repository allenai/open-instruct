# Safety Evaluations

We are using the Ai2 Safety Evaluation suite for safety evals. This contains a bunch of sub-evals, and you can learn more by looking at [the eval-safety fork](https://github.com/nouhadziri/safety-eval-fork).

## Running at Ai2

This should be the most relevant thing for internal Ai2 users of open-instruct. To run evals, use the task suite `SAFETY_EVAL` or `SAFETY_EVAL_REASONING` when calling `submit_eval_jobs.py`. This will create a job that uploads and runs the safety evaluations (and uploads to the leaderboard if the appropriate flag is set).

An example command on a reasoning model would be:
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
      --oe_eval_task_suite "SAFETY_EVAL_REASONING"
```

An example command on a non-reasoning model would be:
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
      --oe_eval_task_suite "SAFETY_EVAL"
```

## Running on an interactive session

Clone [the fork](https://github.com/nouhadziri/safety-eval-fork) and run from that location.

### Safety benchmarks

For all benchmarks requiring safety evaluation unless noted otherwise, as a default, we use the [WildGuard](https://github.com/allenai/wildguard) classifier to evaluate the safety of model outputs.

- [WildGuardTest](https://arxiv.org/abs/2406.18495)
- [Harmbench](https://arxiv.org/abs/2402.04249)
- [ToxiGen](https://arxiv.org/abs/2203.09509)
- [XSTest](https://arxiv.org/abs/2308.01263)
- [JailbreakTrigger (in TrustLLM)](https://arxiv.org/abs/2401.05561)
- [Do-anything-now](https://arxiv.org/abs/2308.03825)
- [WildJailbreak](https://arxiv.org/abs/2406.18510) (both harmful and benign contrast sets)

```commandline
PYTHONPATH=safety-eval python evaluation/run_all_generation_benchmarks.py    \
 --model_name_or_path allenai/tulu-2-dpo-7b     --model_input_template_path_or_name tulu2    \
  --report_output_path ./generation_results/metrics.json     --save_individual_results_path ./generation_results/all.json \
  --hf_upload_name {HF upload name} --upload_to_hf {HF repo ID} --min_gpus_per_task {num. GPUs available}
```

**Changing classifiers for safety benchmarks**:

You can change the safety classifier used for evaluation by specifying the `classifier_model_name` in the yaml file.
For example, when you want to use the HarmBench's classifiers for evaluation on HarmBench, you can use `HarmbenchClassifier` as the `classifier_model_name`. Please check out the `evaluation/tasks/generation/harmbench/default.yaml` and `evaluation/tasks/classification/harmbench/harmbench_classsifier.yaml` to see the classifier's specification.




## Running with gantry

You can also run with gantry, if you want to test changes.
**Important**: Before you run any command with gantry, make sure you *commit and push*, since gantry will attempt to clone the repo with your local latest commit hash.

See the "One-Time Setup" section below before running commands. To test your setup, run the following command -- if this job succeeds, then you're ready to run evaluations with gantry.

```bash
gantry run --workspace {workspace} --budget ai2/oe-adapt --beaker-image kavelr/oe-safety --venv base --cluster ai2/jupiter --env-secret OPENAI_API_KEY=openai_api_key --env-secret HF_TOKEN=hf_token -- python -c 'print("Hello world")'
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
