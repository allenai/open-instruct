"""Count template tokens per parquet row and report a lower bound on packed instances per epoch."""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import transformers
from pyarrow import parquet as pq

TOKENIZER = None


def init_tokenizer(name):
    global TOKENIZER
    TOKENIZER = transformers.PreTrainedTokenizerFast.from_pretrained(name)


def count(path):
    msgs = pq.read_table(path, columns=["messages"]).column(0).to_pylist()
    return [len(TOKENIZER.apply_chat_template(m, tokenize=True, return_dict=False)) for m in msgs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("seq", nargs="?", type=int, default=32768)
    parser.add_argument("tokenizer", nargs="?", default="allenai/olmo-3-tokenizer-instruct-dev")
    parser.add_argument("--output", type=Path, help="Lengths .npy file; defaults to <input_dir>/dolci_lens.npy.")
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    args = parser.parse_args()
    files = sorted(args.input_dir.glob("*.parquet"))
    if not files:
        parser.error("input_dir contains no parquet files")
    if args.seq <= 0 or args.workers <= 0:
        parser.error("seq and workers must be positive")
    output = args.output or args.input_dir / "dolci_lens.npy"
    with Pool(min(args.workers, len(files)), initializer=init_tokenizer, initargs=(args.tokenizer,)) as pool:
        lens = np.concatenate([np.asarray(x, dtype=np.int64) for x in pool.map(count, files)])
    if not len(lens):
        parser.error("input parquet files contain no rows")
    # Refuse to silently replace the lengths from another dataset or tokenizer.
    with output.open("xb") as stream:
        np.save(stream, lens)
    truncated = np.minimum(lens, args.seq)
    print(
        f"rows {len(lens)}  tokens {lens.sum():,}  mean {lens.mean():.0f}  p50 {np.median(lens):.0f}  "
        f"p99 {np.percentile(lens, 99):.0f}  max {lens.max()}  >SEQ {np.mean(lens > args.seq) * 100:.3f}%  "
        f"({(lens - truncated).sum():,} tokens cut)"
    )
    print(
        f"tokens after truncation {truncated.sum():,}; lower bound on {args.seq}-token instances per epoch: "
        f"{int(np.ceil(truncated.sum() / args.seq)):,}"
    )


if __name__ == "__main__":
    main()
