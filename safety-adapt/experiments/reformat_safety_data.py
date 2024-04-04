"""
Reformat training data for open-instruct
"""

import argparse
from pathlib import Path
import random
import json
import os



def tulu_format_safety(args, idx, inst):
    "Put the safety data in tulu format."
    # Keep around the index into the original science dataset.

    return {
        "dataset": args.dataset_desc,
        "id": inst["id"] if "id" in inst else f"{args.dataset_desc}_{idx}",
        "messages": [
            {"role": "user", "content": inst.get("prompt", inst.get("instruction", ""))},
            {"role": "assistant", "content": inst.get("response", inst.get("output", ""))},
        ],
    }


def main():

    parser = argparse.ArgumentParser(
        description="Reformat training safety data for open-instruct.",
    )
    parser.add_argument("--input_file", type=str, help="Instance directory.")
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory. The file name is generated based on the other args.",
    )
    parser.add_argument("--dataset_desc", type=str, default="safety_adapt")
    args = parser.parse_args()

    # load input data
    if args.input_file.endswith("jsonl"):
        data = [json.loads(i) for i in open(args.input_file)]
    elif args.input_file.endswith("json"):
        data = json.load(open(args.input_file))

    # format tulu style
    formatted_data = [tulu_format_safety(args, idx, inst) for idx, inst in enumerate(data)]

    output_file_name = os.path.basename(args.input_file)
    with open(os.path.join(args.out_dir, output_file_name), "w") as f:
        for example in formatted_data:
            f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    main()
