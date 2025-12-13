import argparse
import os
import random

import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flan_v2_data_dir", type=str, default="../open-instruct/data/raw_train/flan_v2")
    parser.add_argument("--total_num_samples", type=int, default=50000)
    parser.add_argument("--output_path", type=str, default="data/raw_train/flan_v2/flan_v2_50k.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mixing_portions", choices=["flan_v1", "flan_v2"], default="flan_v2", help="Mixing weights for flan subsets."
    )
    args = parser.parse_args()
    random.seed(args.seed)

    # The following portions are based on the flan_v2 code: https://github.com/google-research/FLAN/blob/main/flan/v2/run_example.py
    # This is used to build tulu mixture v1.
    portions_v1 = {
        "flan_zsopt": 0.1,
        "flan_fsopt": 0.1,
        "flan_zsnoopt": 0.1,
        "flan_fsnoopt": 0.1,
        "t0_zsopt": 0.08,
        "t0_fsopt": 0.08,
        "t0_zsnoopt": 0.08,
        "t0_fsnoopt": 0.08,
        "niv2_zsopt": 0.1,
        "niv2_fsopt": 0.1,
        "cot_zsopt": 0.025,
        "cot_fsopt": 0.025,
        "dialog_zsopt": 0.015,
        "dialog_fsopt": 0.015,
    }

    # For tulu mixture v2, for only keep the few shot ones since those zero-shot outputs might not be optimal in terms of styles.
    # We also remove dialog since it might be too easy for LLMs.
    portions_v2 = {
        "flan_zsopt": 0,
        "flan_fsopt": 0.2,
        "flan_zsnoopt": 0,
        "flan_fsnoopt": 0.2,
        "t0_zsopt": 0,
        "t0_fsopt": 0.16,
        "t0_zsnoopt": 0,
        "t0_fsnoopt": 0.16,
        "niv2_zsopt": 0,
        "niv2_fsopt": 0.23,
        "cot_zsopt": 0,
        "cot_fsopt": 0.05,
        "dialog_zsopt": 0,
        "dialog_fsopt": 0,
    }

    portions_lookup = {"flan_v1": portions_v1, "flan_v2": portions_v2}
    portions = portions_lookup[args.mixing_portions]

    assert sum(portions.values()) == 1.0

    num_samples = {k: int(v * args.total_num_samples) for k, v in portions.items()}

    with open(args.output_path, "w") as fout:
        for task_name, num_sample in num_samples.items():
            if num_sample == 0:
                continue
            print(f"Sampling {num_sample} samples from {task_name}")
            task_data_path = os.path.join(args.flan_v2_data_dir, task_name, f"{task_name}.jsonl")
            # randomly sample num_sample lines from task_data_path, the data might be very large so we can't load it all into memory
            # we need to first count the total number of lines in the file and then only load the lines we need
            num_lines = 0
            with open(task_data_path) as fin:
                for line in tqdm.tqdm(fin, desc=f"Counting lines in {task_data_path}"):
                    num_lines += 1
            print(f"Sampling {num_sample} lines from {num_lines} lines")
            sampled_lines = random.sample(range(num_lines), num_sample)
            sampled_lines = set(sampled_lines)
            with open(task_data_path) as fin:
                for i, line in tqdm.tqdm(enumerate(fin), desc="Reading the file to save the sampled lines"):
                    if i in sampled_lines:
                        fout.write(line)
