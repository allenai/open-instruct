"""Tokens per row of the shuffled Dolci-Instruct shards under the chat template, and how many 32k packed
instances (first-fit, rows truncated at SEQ) two epochs need; sets the step count of a 2-epoch run."""

import glob
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
import transformers

SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 32768
tok = transformers.PreTrainedTokenizerFast.from_pretrained(
    sys.argv[3] if len(sys.argv) > 3 else "allenai/olmo-3-tokenizer-instruct-dev"
)


def count(f):
    msgs = pq.read_table(f, columns=["messages"]).column(0).to_pylist()
    return [len(tok.apply_chat_template(m, tokenize=True, return_dict=False)) for m in msgs]


if __name__ == "__main__":
    fs = sorted(glob.glob(sys.argv[1] + "/*.parquet"))
    with Pool(min(64, len(fs))) as p:
        lens = np.concatenate([np.array(x) for x in p.map(count, fs)])
    np.save("/opt/scratch/dolci_lens.npy", lens)
    t = np.minimum(lens, SEQ)
    print(
        f"rows {len(lens)}  tokens {lens.sum():,}  mean {lens.mean():.0f}  p50 {np.median(lens):.0f}  p99 {np.percentile(lens, 99):.0f}  max {lens.max()}  >SEQ {np.mean(lens > SEQ) * 100:.3f}%  ({(lens - t).sum():,} tokens cut)"
    )
    print(
        f"tokens after truncation {t.sum():,}; lower bound on {SEQ}-token instances per epoch: {int(np.ceil(t.sum() / SEQ)):,}"
    )
