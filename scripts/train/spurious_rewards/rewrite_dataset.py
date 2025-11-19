dataset_name = "mnoukhov/DAPO-Math-14k-Processed-RLVR"
dataset_name = "allenai/Dolci-RLZero-Math-7B"

from datasets import load_dataset

ds = load_dataset(dataset_name)

print(type(ds["train"]))

def to_random(example):
    example["dataset"] = ["random"]
    return example

ds["train"] = ds["train"].map(to_random)

ds.push_to_hub("stellalisy/Dolci-RLZero-Math-7B_random")

