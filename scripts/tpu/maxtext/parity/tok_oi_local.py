"""Tokenize a stratified sample of local Dolci-Instruct-SFT rows with open-instruct's SFT path.
Strata: tool rows (function_calls / functions / environment), system-prompt rows, multi-turn rows, the rest."""

import glob
import json
import random
import sys

import pyarrow.parquet as pq

from open_instruct import dataset_transformation as dt

per, out = int(sys.argv[1]), sys.argv[2]
P = sorted(glob.glob("/opt/scratch/hf/hub/datasets--allenai--Dolci-Instruct-SFT/snapshots/*/data/*.parquet"))
tok = dt.TokenizerConfig(
    tokenizer_name_or_path="allenai/olmo-3-tokenizer-instruct-dev", chat_template_name="tokenizer_default"
).tokenizer


def stratum(msgs):
    roles = [m["role"] for m in msgs]
    if any(m.get("function_calls") or m.get("functions") for m in msgs) or "environment" in roles:
        return "tool"
    if roles[0] == "system":
        return "system"
    if roles.count("assistant") > 1:
        return "multi"
    return "single"


rng = random.Random(0)
pools = {k: [] for k in ["tool", "system", "multi", "single"]}
for f in P:
    t = pq.read_table(f, columns=["id", "messages"]).to_pylist()
    for row in rng.sample(t, min(len(t), 4 * per)):
        pools[stratum(row["messages"])].append(row)
with open(out, "w") as fo:
    for k, rows in pools.items():
        for row in rng.sample(rows, min(per, len(rows))):
            r = dt.sft_tulu_tokenize_and_truncate_v1({"messages": row["messages"]}, tok, max_seq_length=32768)
            fo.write(
                json.dumps(
                    {
                        "i": row["id"],
                        "stratum": k,
                        "messages": row["messages"],
                        "input_ids": r["input_ids"].tolist(),
                        "labels": r["labels"].tolist(),
                    }
                )
                + "\n"
            )
print({k: min(per, len(v)) for k, v in pools.items()})
