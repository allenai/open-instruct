dataset_name = "hamishivi/rlvr_orz_math_57k_collected"

from datasets import load_dataset

ds = load_dataset(dataset_name)

print(type(ds["train"]))
breakpoint()

def to_random(example):
    example["dataset"][0] = "random"
    return example

ds["train"] = ds["train"].map(to_random)

breakpoint()
ds.push_to_hub("stellalisy/rlvr_orz_math_57k_collected_random")

