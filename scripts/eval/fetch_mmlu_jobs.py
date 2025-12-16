"""
A script for quickly collecting beaker results given a prefix.
Computes mmlu average quckly.
"""

import argparse

from beaker import Beaker
from eval.utils import upload_results_to_hf

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", type=str, required=True)
parser.add_argument("--workspace", type=str, default="ai2/tulu-3-results", help="workspace to search for experiments")
parser.add_argument(
    "--upload_to_hf", type=str, default=None
)  # set to allenai/tulu-3-results//results/<model_name> to upload
args = parser.parse_args()

MMLU_TASKS = [
    "mmlu_abstract_algebra",
    "mmlu_anatomy",
    "mmlu_astronomy",
    "mmlu_business_ethics",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_chemistry",
    "mmlu_college_computer_science",
    "mmlu_college_mathematics",
    "mmlu_college_medicine",
    "mmlu_college_physics",
    "mmlu_computer_security",
    "mmlu_conceptual_physics",
    "mmlu_econometrics",
    "mmlu_electrical_engineering",
    "mmlu_elementary_mathematics",
    "mmlu_formal_logic",
    "mmlu_global_facts",
    "mmlu_high_school_biology",
    "mmlu_high_school_chemistry",
    "mmlu_high_school_computer_science",
    "mmlu_high_school_european_history",
    "mmlu_high_school_geography",
    "mmlu_high_school_government_and_politics",
    "mmlu_high_school_macroeconomics",
    "mmlu_high_school_mathematics",
    "mmlu_high_school_microeconomics",
    "mmlu_high_school_physics",
    "mmlu_high_school_psychology",
    "mmlu_high_school_statistics",
    "mmlu_high_school_us_history",
    "mmlu_high_school_world_history",
    "mmlu_human_aging",
    "mmlu_human_sexuality",
    "mmlu_international_law",
    "mmlu_jurisprudence",
    "mmlu_logical_fallacies",
    "mmlu_machine_learning",
    "mmlu_management",
    "mmlu_marketing",
    "mmlu_medical_genetics",
    "mmlu_miscellaneous",
    "mmlu_moral_disputes",
    "mmlu_moral_scenarios",
    "mmlu_nutrition",
    "mmlu_philosophy",
    "mmlu_prehistory",
    "mmlu_professional_accounting",
    "mmlu_professional_law",
    "mmlu_professional_medicine",
    "mmlu_professional_psychology",
    "mmlu_public_relations",
    "mmlu_security_studies",
    "mmlu_sociology",
    "mmlu_us_foreign_policy",
    "mmlu_virology",
    "mmlu_world_religions",
]

beaker = Beaker.from_env(default_workspace=args.workspace)

# get all experiments in workspace matching the prefix
results = {}
for exp in beaker.workspace.iter_experiments(workspace=args.workspace, match=args.prefix):
    # grab eval name
    eval_name = None
    for name in MMLU_TASKS:
        if name in exp.name:
            eval_name = name + ":mc::tulu"
            break
    if eval_name is None:
        print(f"Experiment {exp.name} does not have a recognized eval name.")
        continue
    # grab metric
    metrics = beaker.experiment.metrics(exp)
    if metrics is None:
        print(f"Experiment {exp.name} has no metrics. Maybe waiting for results?")
        continue
    # grab primary result
    results[eval_name] = metrics["metrics"][0]["primary_score"]


# finally, print results in order given with tab spacing
for eval_name in MMLU_TASKS:
    print(eval_name, "\t", results.get(eval_name + ":mc::tulu", 0))
print()
# compute macro-average (usual mmlu score)
mmlu_sum = 0
for eval_name in MMLU_TASKS:
    mmlu_sum += results.get(eval_name + ":mc::tulu", 0)
mmlu_avg = mmlu_sum / len(MMLU_TASKS)

print(f"Macro-average MMLU: {mmlu_avg}")

results_blob = {"metrics": {"primary_score": mmlu_avg, **results}, "task_name": "mmlu:mc::tulu"}

# upload to huggingface if asked
if args.upload_to_hf is not None:
    hf_datsaset_name = args.upload_to_hf.split("//")[0]
    hf_dataset_save_dir = args.upload_to_hf.split("//")[1]
    upload_results_to_hf(
        results_blob, hf_dataset_name, hf_dataset_save_dir, task_name=None, primary_score=None, prepend_timestamp=False
    )
