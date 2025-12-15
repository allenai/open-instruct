import argparse

import numpy as np
from datasets import Dataset, load_dataset

import open_instruct.utils as open_instruct_utils
from utils import convert_message_keys

"""
Script for converting UltraInteract into standard format.
- Prompt: Text of initial prompt.
- Chosen: Messages list (Dict, role, content format)
- Rejected: Messages list (Dict, role, content format)

Save to hub.
"""


def create_chosen_and_rejected(row):
    """
    Takes in example (dict) and then makes chosen / rejected the "trajectory" key
    with the message from chosen/rejected added to the end (assistant turn)
    """
    chosen = row["trajectory"].copy()
    rejected = row["trajectory"].copy()

    chosen += [{"from": "assistant", "value": row["chosen"]}]
    rejected += [{"from": "assistant", "value": row["rejected"]}]

    row["chosen"] = chosen
    row["rejected"] = rejected

    # delete trajectory
    del row["trajectory"]
    return row


def process_and_upload(task, push_to_hub, hf_entity):
    dataset = load_dataset(
        "openbmb/UltraInteract_pair", split="train", num_proc=open_instruct_utils.max_num_processes()
    )
    dataset_task = dataset.filter(lambda x: x["task"] == task)

    # convert to pandas dataframe
    dataset_df = dataset_task.to_pandas()
    # compute length of trajectory for each row
    dataset_df["length"] = dataset_df["trajectory"].apply(len)

    # take the longest length row per prompt ("parent_id")
    dataset_df_longest = dataset_df.sort_values("length", ascending=False).drop_duplicates("parent_id")

    # convert to messages format (trajectory)
    dataset_hf = Dataset.from_pandas(dataset_df_longest)
    dataset_hf = dataset_hf.map(create_chosen_and_rejected)
    dataset_hf = dataset_hf.map(convert_message_keys)

    # push to hub
    if push_to_hub:
        if hf_entity:
            repo_id = f"{hf_entity}/UltraInteract_pair_maxlen_{task}"
        else:
            raise ValueError("hf_entity must be provided when push_to_hub is True")
        print(f"Pushing dataset to Hub: {repo_id}")
        dataset_hf.push_to_hub(repo_id)
    else:
        print(f"Dataset UltraInteract_pair_maxlen_{task} processed (not pushed to Hub)")

    # alternate dataset with random length per parent id
    dataset_df_random = dataset_df.groupby("parent_id").sample(n=1)

    dataset_hf = Dataset.from_pandas(dataset_df_random)
    dataset_hf = dataset_hf.map(create_chosen_and_rejected)
    dataset_hf = dataset_hf.map(convert_message_keys)

    # push to hub
    if push_to_hub:
        if hf_entity:
            repo_id = f"{hf_entity}/UltraInteract_pair_randomlen_{task}"
        else:
            raise ValueError("hf_entity must be provided when push_to_hub is True")
        print(f"Pushing dataset to Hub: {repo_id}")
        dataset_hf.push_to_hub(repo_id)
    else:
        print(f"Dataset UltraInteract_pair_randomlen_{task} processed (not pushed to Hub)")

    # print length distribution for both datasets
    print(f"Task: {task}")
    print("Longest length dataset:")
    print(np.mean(dataset_df_longest["length"]))
    print(np.std(dataset_df_longest["length"]))
    print("Random length dataset:")
    print(np.mean(dataset_df_random["length"]))
    print(np.std(dataset_df_random["length"]))


def main(push_to_hub: bool, hf_entity: str | None):
    if push_to_hub and hf_entity is None:
        raise ValueError("hf_entity must be provided when push_to_hub is True")

    tasks = ["Coding", "Math_CoT", "Math_PoT", "Logic"]

    for task in tasks:
        process_and_upload(task, push_to_hub, hf_entity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process UltraInteract dataset and optionally upload to Hugging Face Hub."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Upload the dataset to Hugging Face Hub")
    parser.add_argument(
        "--hf_entity",
        type=str,
        default=None,
        help="Hugging Face organization to upload to (must be provided if push_to_hub is True)",
    )

    args = parser.parse_args()

    main(args.push_to_hub, args.hf_entity)
