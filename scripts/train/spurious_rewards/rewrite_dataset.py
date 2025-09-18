dataset_name = "mnoukhov/DAPO-Math-14k-Processed-RLVR"

from datasets import load_dataset

ds = load_dataset(dataset_name)

print(type(ds["train"]))
breakpoint()

def to_random(example):
    example["dataset"] = ["random"]
    return example

ds["train"] = ds["train"].map(to_random)

breakpoint()
ds.push_to_hub("stellalisy/DAPO-Math-14k-Processed-RLVR_random")

