"""Write allenai/Dolci-Think-SFT as globally shuffled parquet shards for MaxText's streaming HF pipeline.

MaxText loads parquet with streaming=True, so its shuffle only permutes shards and mixes a small buffer;
the hub shards are grouped by source, so without a global shuffle each stretch of training would draw from
one source. Keeps every message field (function_calls, functions) for the chat template.
  build_full_dolci.py <out dir> <n shards> <seed>
"""

import glob
import os
import sys

import datasets

src = sorted(glob.glob("/opt/scratch/hf/hub/datasets--allenai--Dolci-Think-SFT/snapshots/*/data/*.parquet"))
out, n_shards, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.makedirs(out, exist_ok=True)
ds = datasets.load_dataset("parquet", data_files=src, split="train", cache_dir="/opt/scratch/hf/datasets")
ds = ds.select_columns([c for c in ("id", "messages", "source_dataset", "source") if c in ds.column_names]).shuffle(seed=seed)
for i in range(n_shards):
    ds.shard(n_shards, i, contiguous=True).to_parquet(f"{out}/train-{i:05d}-of-{n_shards:05d}.parquet")
print(ds.num_rows, "rows ->", n_shards, "shards in", out)
