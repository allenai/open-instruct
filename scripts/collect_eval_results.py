"""
Usage example:
python scripts/collect_eval_results.py \
    --experiment_id 01HV0P4E3MW9211HX0JEKM0PXM \
    --job_suffix _tulu2_13b_dpo_ultrainteract_04082024 \
    --output_file metrics.json \
    --task_order gsm_cot gsm_direct toxigen alpaca_eval \
    --print_table \
    --table_file metrics.tsv
"""

import beaker
from pathlib import Path
import argparse
from copy import deepcopy
import json
import pandas as pd


def make_parser():
    default_task_order = [
        "alpaca_eval",
        "bbh_cot",
        "bbh_direct",
        "codex_eval_temp_0.1",
        "codex_eval_temp_0.8",
        "gsm_cot",
        "gsm_direct",
        "ifeval",
        "mmlu_0shot",
        "mmlu_5shot",
        "toxigen",
        "truthfulqa",
        "tydiqa_goldp_1shot",
        "tydiqa_no_context_1shot",
        "xstest",
    ]

    parser = argparse.ArgumentParser(
        description="""Point this script at a Beaker job created by `submit_eval_jobs.py`.
                    It will will collect all evaluation metrics and dump them in a json
                    file. It will also collect summary metrics for each task.""",
        epilog="""Usage example:
                  python scripts/collect_eval_results.py 
                      --experiment_id 01HV0P4E3MW9211HX0JEKM0PXM 
                      --job_suffix _tulu2_13b_dpo_ultrainteract_04082024 
                      --task_order gsm_cot gsm_direct toxigen alpaca_eval 
                      --output_file metrics.json 
                      --table_file metrics.tsv
                      --print_table""",
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
        "--output_file",
        type=Path,
        default=None,
        help="If given, dump a .json with all metrics to this file.",
    )
    parser.add_argument(
        "--table_file",
        type=Path,
        default=None,
        help="If given, save table of summary results.",
    )
    parser.add_argument(
        "--print_table",
        action="store_true",
        help="If given, print summary metrics to console.",
    )
    parser.add_argument(
        "--task_order",
        nargs="+",
        default=default_task_order,
        help="""If given, display summary metrics in this order.""",
    )

    return parser


def get_summary_metric(metrics, eval_task):
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
        "truthfulqa": ["truth-info acc"],
        "xstest": ["gpt4_label", "refusal_prf", "f1"],
        "alpaca_eval": ["win_rate", "model-greedy-long"],
        "ifeval": ["loose", "Accuracy"],
    }

    # These tasks are scaled 0-100 instead of 0-1; rescale for final summary.
    tasks_scaled_to_100 = [
        "tydiqa_goldp_1shot",
        "tydiqa_no_context_1shot",
        "alpaca_eval",
    ]

    # Get summary metric if available.
    if eval_task not in summary_lookup:
        print(f"No summary metric for {eval_task}. Skipping.")
        return None

    # Descend through result dict to get the result we want.
    key_list = summary_lookup[eval_task]
    result_loop = deepcopy(metrics)
    for key in key_list:
        result_loop = result_loop[key]

    # Rescale the tasks that are on a 0-100 scale.
    if eval_task in tasks_scaled_to_100:
        result_loop = result_loop / 100

    # Return the value and the full field name.
    return result_loop, eval_task + ":" + ":".join(key_list)


def make_summary_table(summary_metrics, args):
    tbl = []
    for k, v in summary_metrics.items():
        splt = k.split(":")
        task = splt[0]
        metric = ":".join(splt[1:])
        tbl.append({"task": task, "metric": metric, "value": v})
    summary_table = pd.DataFrame(tbl).set_index("task")
    summary_table = summary_table.loc[args.task_order].round(4)

    return summary_table


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
        if metrics is None:
            # Case where no metrics are availble.
            continue

        metrics_all[eval_task] = metrics

        summary_metric, full_name = get_summary_metric(metrics, eval_task)
        if summary_metric is not None:
            metrics_summary[full_name] = summary_metric

    metrics_all["summary"] = metrics_summary
    if args.print_table or args.table_file:
        summary_table = make_summary_table(metrics_all["summary"], args)

        if args.print_table:
            print(summary_table)
        if args.table_file is not None:
            summary_table.to_csv(
                args.table_file, sep="\t", index=True, float_format="%0.4f"
            )

    if args.output_file is not None:
        json.dump(metrics_all, open(args.output_file, "w"), indent=2)


if __name__ == "__main__":
    main()
