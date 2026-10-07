"""Like tok_oi.py, but keeps only rows that exercise multi-turn templating: >1 assistant turn, a system
message, or a tool message. Scans the whole Dolci-Instruct-SFT stream until N are collected."""

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
ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True).shuffle(seed=0, buffer_size=20000)
kept = 0
with open(out, "w") as f:
    for i, row in enumerate(ds):
        roles = [m["role"] for m in row["messages"]]
        if not (roles.count("assistant") > 1 or "system" in roles or "tool" in roles):
            continue
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
        kept += 1
        if kept >= N:
            break
print("scanned", i + 1, "kept", kept)
