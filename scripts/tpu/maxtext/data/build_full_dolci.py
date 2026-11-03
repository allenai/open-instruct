"""Globally shuffle one Dolci-Instruct-SFT revision into parquet shards for MaxText."""

import argparse
from pathlib import Path

import datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("n_shards", type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument(
        "--revision", default="main", help="One HF dataset revision; pin a commit for reproducibility."
    )
    parser.add_argument(
        "--data-files", nargs="+", help="Explicit local parquet paths or globs instead of the HF dataset."
    )
    parser.add_argument("--cache-dir", help="Datasets cache; otherwise use the standard HF cache settings.")
    args = parser.parse_args()
    if args.n_shards <= 0:
        parser.error("n_shards must be positive")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error("out must be a new or empty directory; old shards would be mixed into training")

    if args.data_files:
        ds = datasets.load_dataset("parquet", data_files=args.data_files, split="train", cache_dir=args.cache_dir)
    else:
        # Loading the dataset ID resolves one revision, rather than globbing every cached snapshot.
        ds = datasets.load_dataset(
            "allenai/Dolci-Instruct-SFT", revision=args.revision, split="train", cache_dir=args.cache_dir
        )
    if args.n_shards > ds.num_rows:
        parser.error(f"n_shards ({args.n_shards}) exceeds the dataset's {ds.num_rows} rows")
    ds = ds.select_columns(["id", "messages", "source_dataset"]).shuffle(seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n_shards):
        ds.shard(args.n_shards, i, contiguous=True).to_parquet(
            args.out / f"train-{i:05d}-of-{args.n_shards:05d}.parquet"
        )
    print(ds.num_rows, "rows ->", args.n_shards, "shards in", args.out)


if __name__ == "__main__":
    main()
