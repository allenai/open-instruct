"""Real-data check of a replay dir written by scripts/data/mask_replay_final_eos.py (ledger H270's fix, GPU path).

    python scripts/data/check_replay_eos_mask.py --replay_dir SRC --masked_dir OUT \\
        [--chat_dir CHAT_CACHE ...] [--max_seq_length 65536] [--out check.json]

It uses numpy and the standard library only, and none of the code that wrote OUT. Replay documents are
identified by where they come from: every `.csv.gz` row of SRC is one. For each part:
  1. OUT's token part and `.csv.gz` are SRC's (the same file, or the same sha256), and the rows tile it;
  2. no label is added, and the labels cleared (trained in SRC, untrained in OUT) are exactly the final
     token of each replay document that has >= 2 tokens, ends in <|endoftext|> and is trained in SRC;
  3. no replay document starts with <|im_start|> and every one ends in <|endoftext|>, as C16's replay
     dirs are built, so the patch's rule and the source agree on which documents are replay.
Each --chat_dir (the chat cache trained beside it, which the fix leaves unmodified) must have every
document start with <|im_start|>, so the rule would clear nothing there either. Replay documents longer
than --max_seq_length are counted: training truncates them, so their cleared label never reaches a batch.
PASS (exit 0) iff every zero count is 0, replay documents exist, and the labels cleared equal the replay
documents with a trained final <|endoftext|>.
"""

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import sys

import numpy as np

EOS, IM_START = 100257, 100264
TOKEN_PART = re.compile(r"token_ids_part_(\d+)\.npy$")
ZERO = (
    "part_sets_differ",
    "untiled_parts",
    "token_or_boundary_files_differ",
    "mask_length_mismatch",
    "mask_not_binary",
    "labels_added",
    "dropped_elsewhere",
    "replay_final_kept",
    "replay_starting_im_start",
    "replay_not_ending_eos",
    "chat_not_im_start",
)


def sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def token_parts(numpy_dir: str) -> dict[str, str]:
    return {m.group(1): m.group(0) for m in map(TOKEN_PART.match, os.listdir(numpy_dir)) if m}


def load_part(numpy_dir: str, index: str):
    """(tokens, rows, tiled) of one part: rows are its `.csv.gz` (start, end) lines."""
    num_tokens = os.path.getsize(os.path.join(numpy_dir, f"labels_mask_part_{index}.npy"))
    token_path = os.path.join(numpy_dir, f"token_ids_part_{index}.npy")
    itemsize = os.path.getsize(token_path) // max(num_tokens, 1)
    tokens = np.memmap(token_path, mode="r", dtype=np.dtype(f"uint{8 * itemsize}"))
    with gzip.open(os.path.join(numpy_dir, f"token_ids_part_{index}.csv.gz"), "rt") as f:
        rows = np.array([line.split(",")[:2] for line in f if line.strip()], dtype=np.int64).reshape(-1, 2)
    tiled = (
        len(rows) > 0
        and len(tokens) == num_tokens
        and rows[0, 0] == 0
        and rows[-1, 1] == num_tokens
        and bool((rows[1:, 0] == rows[:-1, 1]).all())
        and bool((rows[:, 1] >= rows[:, 0]).all())
    )
    return tokens, rows[rows[:, 1] > rows[:, 0]], tiled


def check(
    replay_dir: str,
    masked_dir: str,
    chat_dirs: list[str],
    max_seq_length: int = 65536,
    eos: int = EOS,
    im_start: int = IM_START,
) -> dict:
    t: collections.Counter = collections.Counter()
    per_part = []
    src_parts, out_parts = token_parts(replay_dir), token_parts(masked_dir)
    t["part_sets_differ"] = int(sorted(src_parts) != sorted(out_parts))
    for index in sorted(set(src_parts) & set(out_parts)):
        tokens, rows, tiled = load_part(replay_dir, index)
        if not tiled:
            t["untiled_parts"] += 1
            continue
        for name in (f"token_ids_part_{index}.npy", f"token_ids_part_{index}.csv.gz"):
            a, b = os.path.join(replay_dir, name), os.path.join(masked_dir, name)
            t["token_or_boundary_files_differ"] += os.path.realpath(a) != os.path.realpath(b) and sha256(a) != sha256(
                b
            )
        src = np.fromfile(os.path.join(replay_dir, f"labels_mask_part_{index}.npy"), dtype=np.uint8)
        out = np.fromfile(os.path.join(masked_dir, f"labels_mask_part_{index}.npy"), dtype=np.uint8)
        if src.shape != out.shape:
            t["mask_length_mismatch"] += 1
            continue
        t["mask_not_binary"] += int((src > 1).sum() + (out > 1).sum())
        src, out = src != 0, out != 0
        t["labels_added"] += int((~src & out).sum())
        dropped = np.flatnonzero(src & ~out)
        lengths, first, last = rows[:, 1] - rows[:, 0], tokens[rows[:, 0]], tokens[rows[:, 1] - 1]
        final = rows[:, 1] - 1
        ends_in_eos = (lengths >= 2) & (last == eos)
        expected = final[ends_in_eos & src[final]]
        hit = np.isin(dropped, expected)
        part = {
            "replay_docs": len(rows),
            "replay_tokens": int(lengths.sum()),
            "replay_starting_im_start": int((first == im_start).sum()),
            "replay_not_ending_eos": int((last != eos).sum()),
            "replay_one_token": int((lengths < 2).sum()),
            "replay_final_untrained_in_source": int((ends_in_eos & ~src[final]).sum()),
            "replay_over_max_seq_length": int((lengths > max_seq_length).sum()),
            "expected_drops": len(expected),
            "dropped": len(dropped),
            "dropped_at_replay_final": int(hit.sum()),
            "dropped_elsewhere": int((~hit).sum()),
            "replay_final_kept": int((~np.isin(expected, dropped)).sum()),
        }
        t.update(part)
        per_part.append({"part": index, **part})
    for chat_dir in chat_dirs:
        for index in sorted(token_parts(chat_dir)):
            tokens, rows, tiled = load_part(chat_dir, index)
            if not tiled:
                t["untiled_parts"] += 1
                continue
            not_chat = tokens[rows[:, 0]] != im_start
            t["chat_docs"] += len(rows)
            t["chat_not_im_start"] += int(not_chat.sum())
            t["chat_rule_would_drop"] += int(
                (not_chat & (rows[:, 1] - rows[:, 0] >= 2) & (tokens[rows[:, 1] - 1] == eos)).sum()
            )
    ok = (
        all(t[k] == 0 for k in ZERO)
        and t["expected_drops"] > 0
        and t["dropped"] == t["expected_drops"] == t["dropped_at_replay_final"]
    )
    return {
        "pass": bool(ok),
        "totals": {k: int(v) for k, v in t.items()},
        "zero_checks": list(ZERO),
        "replay_dir": os.path.realpath(replay_dir),
        "masked_dir": os.path.realpath(masked_dir),
        "chat_dirs": [os.path.realpath(d) for d in chat_dirs],
        "eos_token_id": eos,
        "im_start_token_id": im_start,
        "max_seq_length": max_seq_length,
        "check_sha256": sha256(__file__),
        "per_part": per_part,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay_dir", required=True, help="The source replay dir (unmasked).")
    parser.add_argument("--masked_dir", required=True, help="mask_replay_final_eos.py's output for it.")
    parser.add_argument("--chat_dir", action="append", default=[], help="Chat cache trained beside it; repeatable.")
    parser.add_argument("--max_seq_length", type=int, default=65536)
    parser.add_argument("--eos_token_id", type=int, default=EOS)
    parser.add_argument("--im_start_token_id", type=int, default=IM_START)
    parser.add_argument("--out", help="Write the full result, with per-part counts, here.")
    args = parser.parse_args(argv)
    result = check(
        args.replay_dir, args.masked_dir, args.chat_dir, args.max_seq_length, args.eos_token_id, args.im_start_token_id
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "per_part"}, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
