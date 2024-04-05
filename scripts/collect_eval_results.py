"""
Usage example:
python scripts/collect_eval_results.py \
    --experiment_id 01HTNY22Z9YJZ6HDQBF182BGE6 \
    --job_suffix _v0-step_1.5T-warmup_true-steps_50B_04052024 \
    --output_file metrics.json
"""

import beaker
from pathlib import Path
import argparse
from copy import deepcopy
import json


def make_parser():
    parser = argparse.ArgumentParser(
        description="""Point this script at a Beaker job created by `submit_eval_jobs.py`.
                    It will will collect all evaluation metrics and dump them in a json
                    file. It will also collect summary metrics for each task."""
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        help="ID of the Beaker experiment for which to collect metrics.",
    )
    parser.add_argument(
        "--job_suffix",
        type=str,
        help="Suffix of job name; remove this from the job name to get the eval task.",
    )
    parser.add_argument(
        "--output_file", type=Path, help="Output file for collected results."
    )

    return parser


def get_summary_metric(metrics, eval_task):
    # TODO we need summary metrics for XSTest and AlpacaEval
    summary_lookup = {
        "mmlu_0shot": ["average_acc"],
        "mmlu_5shot": ["average_acc"],
        "gsm_direct": ["exact_match"],
        "gsm_cot": ["exact_match"],
        "bbh_direct": ["average_exact_match"],
        "bbh_cot": ["average_exact_match"],
        "codex_eval_temp_0.1": ["pass@1"],
        "codex_eval_temp_0.8": ["pass@10"],
        "tydiqa_no_context_1shot": ["average", "f1"],
        "tydiqa_goldp_1shot": ["average", "f1"],
        "toxigen": ["overall"],
        "trutufulqa": ["truth-info acc"],
    }

    # Get summary metric if available.
    if eval_task not in summary_lookup:
        print(f"No summary metric for {eval_task}. Skipping.")
        return None

    # Descend through result dict to get the result we want.
    key_list = summary_lookup[eval_task]
    result_loop = deepcopy(metrics)
    for key in key_list:
        result_loop = result_loop[key]

    return result_loop


def main():
    parser = make_parser()
    args = parser.parse_args()

    b = beaker.Beaker.from_env()
    experiment = b.experiment.get(experiment=args.experiment_id)

    metrics_all = {}
    metrics_summary = {}

    for job in experiment.jobs:
        # Skip unfinished jobs.
        eval_task = job.name.replace(args.job_suffix, "").replace(
            "open_instruct_eval_", ""
        )

        if not job.is_done:
            print(f"Eval for {eval_task} not finished. Skipping.")
            continue

        metrics = b.job.metrics(job.id)
        metrics_all[eval_task] = metrics

        summary_metric = get_summary_metric(metrics, eval_task)
        if summary_metric is not None:
            metrics_summary[eval_task] = summary_metric

    metrics_all["summary"] = metrics_summary

    json.dump(metrics_all, open(args.output_file, "w"), indent=2)


if __name__ == "__main__":
    main()
