"""
Script for converting UltraInteract into standard format.
- Prompt: Text of initial prompt.
- Chosen: Messages list (Dict, role, content format)
- Rejected: Messages list (Dict, role, content format)

Save to hub.
"""

from datasets import load_dataset, Dataset
import numpy as np
from utils import convert_message_keys

tasks = ["Coding", "Math_CoT", "Math_PoT", "Logic"]

dataset = load_dataset("openbmb/UltraInteract_pair", split="train")

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

for task in tasks:
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
    dataset_hf.push_to_hub(f"ai2-adapt-dev/UltraInteract_pair_maxlen_{task}")

    # alternate dataset with random length per parent id
    dataset_df_random = dataset_df.groupby("parent_id").sample(n=1)

    dataset_hf = Dataset.from_pandas(dataset_df_random)
    dataset_hf = dataset_hf.map(create_chosen_and_rejected)
    dataset_hf = dataset_hf.map(convert_message_keys)

    # push to hub
    dataset_hf.push_to_hub(f"ai2-adapt-dev/UltraInteract_pair_randomlen_{task}")

    # print length distribution for both datasets
    print(f"Task: {task}")
    print("Longest length dataset:")
    print(np.mean(dataset_df_longest["length"]))
    print(np.std(dataset_df_longest["length"]))
    print("Random length dataset:")
    print(np.mean(dataset_df_random["length"]))
    print(np.std(dataset_df_random["length"]))