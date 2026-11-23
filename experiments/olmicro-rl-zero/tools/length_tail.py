"""Where do the tokens go? Length buckets, token share, correctness and repetition for sweep raw shards.

Decides whether a lower output cap (e.g. 16k) loses anything: a response longer than the cap would have been
truncated (reward 0) under it, while shorter responses are unchanged.

Usage: PYTHONPATH=/path/to/open-instruct python length_tail.py SWEEP_DIR [CAP_TO_TEST ...]
"""

import glob
import json
import sys
import zlib
from multiprocessing import Pool

from open_instruct.ground_truth_utils import MathVerifier

VERIFIER = MathVerifier()
BUCKETS = [(0, 4096), (4096, 8192), (8192, 16384), (16384, 32768), (32768, 10**9)]


def check(item):
    text, label, truncated = item
    tail = text[-4000:].encode()
    loop = len(zlib.compress(tail)) / max(1, len(tail)) < 0.15
    return (0.0 if truncated else VERIFIER([], text, label).score), loop


def main():
    root, caps = sys.argv[1], [int(c) for c in sys.argv[2:]] or [16384]
    items, tokens, files = [], [], sorted(glob.glob(f"{root}/*.raw.json"))
    for path in files:
        for prompt in json.load(open(path))["prompts"]:
            for r in prompt["responses"]:
                truncated = (r["finish"] or {}).get("type") == "length"
                items.append((r["text"], prompt["label"], truncated))
                tokens.append((r["tokens"], truncated))
    with Pool(160) as pool:
        results = pool.map(check, items, chunksize=8)
    total = sum(t for t, _ in tokens)
    print(f"shards: {len(files)}  responses: {len(tokens):,}  tokens: {total / 1e6:.1f}M  overall correct: {sum(s for s, _ in results) / len(results):.4f}")
    for lo, hi in BUCKETS:
        idx = [i for i, (t, _) in enumerate(tokens) if lo < t <= hi or (lo == 0 and t == 0)]
        if not idx:
            continue
        toks = sum(tokens[i][0] for i in idx)
        label = f"{lo // 1024}k-{hi // 1024}k" if hi < 10**9 else f">{lo // 1024}k"
        print(f"  {label:>8}: {len(idx) / len(tokens):7.2%} of responses, {toks / total:6.1%} of tokens, "
              f"correct {sum(results[i][0] for i in idx) / len(idx):.3f}, truncated {sum(tokens[i][1] for i in idx) / len(idx):.2%}, "
              f"repetition-loop tails {sum(results[i][1] for i in idx) / len(idx):.1%}")
    for cap in caps:
        over = [i for i, (t, _) in enumerate(tokens) if t > cap]
        lost = sum(results[i][0] for i in over)
        print(f"cap {cap:,}: {len(over):,} responses exceed it ({len(over) / len(tokens):.2%}); correct answers lost: {lost:.0f} "
              f"({lost / max(1, sum(s for s, _ in results)):.2%} of all correct); tokens saved ≈ "
              f"{sum(tokens[i][0] - cap for i in over) / total:.1%}")


if __name__ == "__main__":
    main()
