"""
Mix together all datasets to create instruction tuning mix.
"""

import json
import os
from pathlib import Path


def write_jsonl(xs, fname):
    with open(fname, "w") as f:
        for x in xs:
            print(json.dumps(x), file=f)


def load_jsonl(fname):
    with open(fname) as f:
        return [json.loads(line) for line in f]


names = ["evidence_inference", "qasper_truncated_4000", "scifact_json", "scitldr_aic", "scierc_ner", "scierc_relation"]

# This is an instruction dataset about several science tasks that David and some other collaborators created.
# Please contact us if you want to use the raw files
data_dir = Path("../../davidw/proj/science-instruct/promptsource-sciit/prompts_davidw/tasks")
out_dir = Path("data/raw_train/science")
os.makedirs(out_dir, exist_ok=True)

full_dataset = []

for name in names:
    ds = load_jsonl(data_dir / f"{name}_train.jsonl")
    for entry in ds:
        entry["dataset"] = name
        full_dataset.append(entry)

write_jsonl(full_dataset, out_dir / "science_train.jsonl")
