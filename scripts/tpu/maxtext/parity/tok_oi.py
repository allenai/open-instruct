"""Tokenize Dolci-Instruct-SFT rows with open-instruct's SFT path; dump input_ids + labels."""

import json
import sys

from datasets import load_dataset

from open_instruct import dataset_transformation as dt

N = int(sys.argv[1])
out = sys.argv[2]
tc = dt.TokenizerConfig(
    tokenizer_name_or_path="allenai/olmo-3-tokenizer-instruct-dev", chat_template_name="tokenizer_default"
)
tok = tc.tokenizer
ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
with open(out, "w") as f:
    for i, row in enumerate(ds):
        if i >= N:
            break
        r = dt.sft_tulu_tokenize_and_truncate_v1({"messages": row["messages"]}, tok, max_seq_length=32768)
        f.write(
            json.dumps(
                {
                    "i": i,
                    "id": row.get("id"),
                    "messages": row["messages"],
                    "input_ids": r["input_ids"].tolist(),
                    "labels": r["labels"].tolist(),
                }
            )
            + "\n"
        )
