"""
Create training mixture from Tulu and science data.
"""

import datasets
import argparse
from pathlib import Path
import random
import json

# from lib import paths
# from lib import util


def get_tulu(args, n_science_insts):
    "Get the tulu dataset, but filter out the science data."
    # If no tulu data, return None.
    if args.tulu == "none":
        return []

    tulu = datasets.load_dataset("allenai/tulu-v2-sft-mixture")

    # Filter out science instances, which are part of test set.
    # tulu_insts = []
    # for entry in tulu["train"]:
    #     if entry["dataset"].split(".")[0] != "science":
    #         tulu_insts.append(entry)

    # Faeze: get all tulu
    tulu_insts = []
    for entry in tulu["train"]:
        tulu_insts.append(entry)

    random.shuffle(tulu_insts)
    if args.tulu == "all":
        return tulu_insts
    else:
        return tulu_insts[:n_science_insts]


def tulu_format_science(inst):
    "Put the science data in tulu format."
    # Keep around the index into the original science dataset.
    info = inst["_instance_id"].split(":")
    task_name = info[0]
    instance_index = info[2]

    return {
        "dataset": f"science.{task_name}",
        "id": f"science.{task_name}.{instance_index}",
        "messages": [
            {"role": "user", "content": inst["input"]},
            {"role": "assistant", "content": inst["output"]},
        ],
    }


# def get_eval_tasks():
#     "Get evaluation tasks."
#     # TODO(davidw) Put the eval tasks in a yaml somewhere.
#     eval_task_dir = paths.EVAL_DIR / "eleuther_templates/tulu"
#     names = [p.stem for p in eval_task_dir.glob("*.yaml")]
#     names = [name for name in names if name != "_default_template"]

#     return names


# def get_one_science_task(args, task_dir):
#     "Format instances for a single science task."
#     insts = util.load_jsonl(task_dir / "train.jsonl")[: args.instances_per_task]

#     return [tulu_format_science(inst) for inst in insts]


# def get_science_tasks(args):
#     "Format all science tasks."
#     if args.instances_per_task == 0:
#         return []

#     eval_tasks = get_eval_tasks()
#     inst_dir = Path(args.instance_dir)
#     res = []
#     for task_dir in inst_dir.iterdir():
#         # Skip eval tasks unless asked to include.
#         if not args.include_eval_tasks and task_dir.name in eval_tasks:
#             continue

#         task_insts = get_one_science_task(args, task_dir)
#         res.extend(task_insts)

#     return res

def get_safety(args):
    "Format all safety tasks."
    # if args.instances_per_task == 0:
    #     return []

    inst_dir = Path(args.instance_dir)
    res = [json.loads(i) for i in open(inst_dir)]

    return res


# def create_mixture(args):
#     "Create train mixture."
#     random.seed(76)  # For data reproducibility.

#     # Skip data that's already created.
#     eval_str = "yes" if args.include_eval_tasks else "no"
#     out_name = f"tulu_{args.tulu}_science_{args.instances_per_task}_eval_{eval_str}"
#     out_dir = Path(args.out_dir)
#     out_file = out_dir / f"{out_name}.jsonl"

#     if out_file.exists():
#         print(f"{out_name} already exists; skipping.")
#         return

#     print(f"Making {out_name}.")

#     science = get_science_tasks(args)
#     tulu = get_tulu(args, len(science))

#     full_data = science + tulu
#     random.shuffle(full_data)

#     util.write_jsonl(full_data, out_file)


def create_safety_tulu_mixture(args):
    "Create safety_tulu train mixture."
    random.seed(76)  # For data reproducibility.


    # Skip data that's already created.
    out_name = f"tulu_{args.tulu}_safety"
    out_dir = Path(args.out_dir)
    out_file = out_dir / f"{out_name}.jsonl"

    if out_file.exists():
        print(f"{out_name} already exists; skipping.")
        return

    print(f"Making {out_name}.")

    safety = get_safety(args)
    tulu = get_tulu(args, len(safety))

    full_data = safety + tulu
    random.shuffle(full_data)

    with open(out_file, "w") as f:
        for example in full_data:
            f.write(json.dumps(example) + "\n")
    # util.write_jsonl(full_data, out_file)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Create training mixture from Tulu and science/safety data.",
    )
    parser.add_argument(
        "--instances_per_task", type=int, help="Number of instances per task.",
    )
    parser.add_argument("--instance_dir", type=str, help="Instance directory.")
    parser.add_argument(
        "--tulu",
        choices=["all", "match", "none"],
        help="How much tulu data. `all` uses all of it, `match` uses as much as science, `none` uses none.",
    )
    parser.add_argument(
        "--include_eval_tasks",
        action="store_true",
        help="If given, include evaluation tasks in mix.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory. The file name is generated based on the other args.",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    # create_mixture(args)
    create_safety_tulu_mixture(args)


if __name__ == "__main__":
    main()
