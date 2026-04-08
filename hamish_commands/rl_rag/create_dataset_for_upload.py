from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import HfApi
import json

# --------------------------------------------------
# config: dataset names and repetition factors
# --------------------------------------------------
datasets = {
    "rl-rag/rl_rag_train_sqa_1k_clean_search_rubric_longform_rubrics_adaptive_rubric": 1.0,
    "rl-rag/rl_rag_train_os_0915_2k_search_rubric_longform_rubrics_adaptive_rubric": 1.0,
    "rl-rag/rl_rag_train_sa_3k_longform_rubrics_adaptive_rubric": 1.0,
    # "rl-rag/RaR-Medicine-20k-o3-mini-converted": 3000,
    # "rl-rag/RaR-Science-20k-o3-mini-converted": 1000,
}

# --------------------------------------------------
# load + expand
# --------------------------------------------------
expanded_splits = {}  # split_name -> list[Dataset]

for ds_name, size in datasets.items():
    ds = load_dataset(ds_name, split="train")
    if isinstance(size, float):
        size = int(size * len(ds))
    elif isinstance(size, int):
        pass
    else:
        raise ValueError(f"Invalid size: {size}")
    # no shuffling.
    ds = ds.select(range(size))
    expanded_splits[ds_name] = ds

# --------------------------------------------------
# concatenate per split
# --------------------------------------------------
final = {}
final_ds = concatenate_datasets(expanded_splits.values())

# --------------------------------------------------
# sanity check
# --------------------------------------------------
print(final_ds, len(final_ds))

# --------------------------------------------------
# upload to HF
# --------------------------------------------------
TARGET_REPO = "rl-research/dr-tulu-rl-data"

# push
final_ds.push_to_hub(TARGET_REPO)
print("Uploaded to", TARGET_REPO)