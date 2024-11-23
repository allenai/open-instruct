# Safety Evaluations

We are using the Ai2 Safety Evaluation suite for safety evals. This contains a bunch of sub-evals, and you can learn more by looking at [the eval-safety fork](https://github.com/nouhadziri/safety-eval-fork).

## Running at Ai2

This should be the most relevant thing for internal Ai2 users of open-instruct. To run evals, simply add `--run_safety_evaluations` when calling `submit_eval_jobs.py`. This will auto-add a job that uploads and runs the safety evaluations (and uploads to the leaderboard if the appropriate flag is set). This uses the `hamishivi/safety-eval` image, which is build from [the eval-safety fork](https://github.com/nouhadziri/safety-eval-fork).

An example command would be:
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
      --run_safety_evaluations
```

Use the `--use_alternate_safety_image` to change the safety image, for example: `--use_alternate_safety_image hamishivi/safety_eval_olmo`.

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



