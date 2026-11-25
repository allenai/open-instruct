"""Real-data check of a replay dir written by scripts/data/mask_replay_final_eos.py (ledger H270's fix, GPU path).

    python scripts/data/check_replay_eos_mask.py --replay_dir SRC --masked_dir OUT \\
        [--pins h182 | --pins_json PINS.json] [--chat_dir CHAT_CACHE ...] [--max_seq_length 65536] [--out check.json]

It uses numpy and the standard library only, and none of the code that wrote OUT. Replay documents are found from
the replay build's inventory, not from the rule under test: SRC's manifest.jsonl, written by the ledger's
experiments/c11/c11_replay.py build, pinned by its sha256 and by the ids, row and token counts of the build's R2
receipt. For H182's replay dir (--pins h182, the default) the counts and ids_sha256 are
experiments/h182/receipts/r2_b1c12r.json's; the manifest sha256 was computed from WEKA on 2026-11-24 and is not
yet recorded in the ledger. The check:
  1. SRC and OUT each hold token, mask and .csv.gz parts with one shared index set, equal to the manifest's parts,
     and no other part-named file; OUT's token parts, .csv.gz parts and manifest are SRC's (same file or sha256).
  2. Each part's manifest rows tile it and equal its .csv.gz rows (the boundaries training reads), so no document
     is cut between parts.
  3. Every inventory document is found exactly once, at its manifest row: its length is length_with_eos, its last
     token is <|endoftext|>, and the tokens before it hash to its tokens_sha256.
  4. Every inventory document's final label is untrained in OUT: a drop if SRC trained it, counted separately as
     already untrained if not. No other label differs: none cleared elsewhere, none added.
  5. Every .csv.gz line is a start,end row, and OUT's dataset_statistics.json passes main's startup reading of it:
     the row-aligned marker, total_instances equal to the metadata line count, and trainable_tokens equal to
     OUT's trained labels.
Each --chat_dir (the chat cache trained beside it, which the fix leaves unmodified) must have rows that tile each
part and every document starting with <|im_start|>, so H270's rule would clear nothing there either. Inventory
documents longer than --max_seq_length are counted: training truncates them, so their cleared label never reaches
a batch.
PASS (exit 0) iff every zero count is 0 and found == drops + already untrained == the pinned row count.
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
MANIFEST = "manifest.jsonl"
PINS = {
    "h182": {
        "manifest_sha256": "2683f40911e79e95bae4d2202ada1b7f677b2b8daacf649b25ee929cd8dcc412",
        "ids_sha256": "2ca4b25da231c5c8262faac6515577ac97c17c9338e49d656c0e73666f2e5aa8",
        "rows": 58480,
        "tokens": 2678796581,
    }
}
PART_NAME = re.compile(r"(token_ids_part_|labels_mask_part_)(\d+)(\.npy|\.csv\.gz)")
PART_KINDS = {
    ("token_ids_part_", ".npy"): "tokens",
    ("labels_mask_part_", ".npy"): "mask",
    ("token_ids_part_", ".csv.gz"): "metadata",
}
ZERO = (
    "manifest_sha256_mismatch",
    "ids_sha256_mismatch",
    "rows_mismatch",
    "tokens_mismatch",
    "duplicate_ids",
    "part_set_mismatch",
    "orphan_part_files",
    "files_differ",
    "untiled_parts",
    "boundary_rows_differ",
    "metadata_malformed_lines",
    "statistics_missing",
    "statistics_mismatch",
    "not_found",
    "mask_not_binary",
    "labels_added",
    "cleared_elsewhere",
    "final_still_trained",
    "inventory_starting_im_start",
    "chat_not_im_start",
)


def sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def list_parts(numpy_dir: str) -> tuple[dict[str, set[str]], list[str]]:
    """({kind: part indices}, orphans): every file named like a part must be an indexed token, mask or .csv.gz."""
    found: dict[str, set[str]] = {kind: set() for kind in PART_KINDS.values()}
    orphans = []
    for name in sorted(os.listdir(numpy_dir)):
        if not name.startswith(("token_ids_part_", "labels_mask_part_")):
            continue
        match = PART_NAME.fullmatch(name)
        kind = PART_KINDS.get((match.group(1), match.group(3))) if match else None
        if match is None or kind is None:
            orphans.append(name)
        else:
            found[kind].add(match.group(2))
    return found, orphans


def same_file(a: str, b: str) -> bool:
    return os.path.exists(b) and (os.path.realpath(a) == os.path.realpath(b) or sha256(a) == sha256(b))


def token_memmap(numpy_dir: str, index: str, num_tokens: int) -> np.memmap:
    path = os.path.join(numpy_dir, f"token_ids_part_{index}.npy")
    return np.memmap(path, mode="r", dtype=np.dtype(f"uint{8 * (os.path.getsize(path) // num_tokens)}"))


def csv_rows(numpy_dir: str, index: str) -> tuple[np.ndarray, int]:
    """A part's `.csv.gz` as (start, end) rows, and how many of its lines are not such a row: OLMo-core cannot
    parse those, and main's startup check counts every line as a row."""
    rows, malformed = [], 0
    with gzip.open(os.path.join(numpy_dir, f"token_ids_part_{index}.csv.gz"), "rt") as f:
        for line in f:
            fields = line.rstrip("\n").split(",")
            if len(fields) >= 2 and fields[0].isdigit() and fields[1].isdigit():
                rows.append((int(fields[0]), int(fields[1])))
            else:
                malformed += 1
    return np.array(rows, dtype=np.int64).reshape(-1, 2), malformed


def tiles(rows: np.ndarray, num_tokens: int) -> bool:
    return bool(
        len(rows)
        and rows[0, 0] == 0
        and rows[-1, 1] == num_tokens
        and (rows[1:, 0] == rows[:-1, 1]).all()
        and (rows[:, 1] >= rows[:, 0]).all()
    )


def check(
    replay_dir: str,
    masked_dir: str,
    pins: dict,
    chat_dirs: list[str] = (),
    max_seq_length: int = 65536,
    eos: int = EOS,
    im_start: int = IM_START,
) -> dict:
    t: collections.Counter = collections.Counter()
    manifest_path = os.path.join(replay_dir, MANIFEST)
    t["manifest_sha256_mismatch"] = int(sha256(manifest_path) != pins["manifest_sha256"])
    with open(manifest_path) as f:
        inventory = [json.loads(line) for line in f if line.strip()]
    ids = [r["id"] for r in inventory]
    t["ids_sha256_mismatch"] = int(hashlib.sha256(json.dumps(ids).encode()).hexdigest() != pins["ids_sha256"])
    t["rows_mismatch"] = int(len(inventory) != pins["rows"])
    t["tokens_mismatch"] = int(sum(r["end"] - r["start"] for r in inventory) != pins["tokens"])
    t["duplicate_ids"] = len(ids) - len(set(ids))
    t["files_differ"] += int(not same_file(manifest_path, os.path.join(masked_dir, MANIFEST)))
    by_part = collections.defaultdict(list)
    for r in inventory:
        by_part[f"{int(r['part']):04d}"].append(r)
    for numpy_dir in (replay_dir, masked_dir):
        found, orphans = list_parts(numpy_dir)
        t["orphan_part_files"] += len(orphans)
        t["part_set_mismatch"] += int(not (found["tokens"] == found["mask"] == found["metadata"] == set(by_part)))

    per_part = []
    for index in sorted(by_part):
        names = [f"{prefix}{index}{ext}" for prefix, ext in PART_KINDS]
        if not all(os.path.exists(os.path.join(d, n)) for d in (replay_dir, masked_dir) for n in names):
            continue  # counted by part_set_mismatch
        for name in (f"token_ids_part_{index}.npy", f"token_ids_part_{index}.csv.gz"):
            t["files_differ"] += int(not same_file(os.path.join(replay_dir, name), os.path.join(masked_dir, name)))
        num_tokens = os.path.getsize(os.path.join(replay_dir, f"labels_mask_part_{index}.npy"))
        tokens = token_memmap(replay_dir, index, num_tokens)
        records = by_part[index]
        rows = np.array([(r["start"], r["end"]) for r in records], dtype=np.int64)
        starts, ends = rows[:, 0], rows[:, 1]
        if not (
            len(tokens) == num_tokens == os.path.getsize(os.path.join(masked_dir, f"labels_mask_part_{index}.npy"))
            and starts[0] == 0
            and ends[-1] == num_tokens
            and (starts[1:] == ends[:-1]).all()
            and (ends > starts).all()
        ):
            t["untiled_parts"] += 1
            continue
        metadata, malformed = csv_rows(replay_dir, index)
        t["metadata_malformed_lines"] += malformed
        t["boundary_rows_differ"] += int(not np.array_equal(metadata, rows))
        part: collections.Counter = collections.Counter()
        for r, s, e in zip(records, starts, ends):
            doc = np.asarray(tokens[s:e])
            found = (
                e - s == r.get("length_with_eos", e - s)
                and doc[-1] == eos
                and hashlib.sha256(doc[:-1].astype("<u4").tobytes()).hexdigest() == r["tokens_sha256"]
            )
            part["found" if found else "not_found"] += 1
        src = np.fromfile(os.path.join(replay_dir, f"labels_mask_part_{index}.npy"), dtype=np.uint8)
        out = np.fromfile(os.path.join(masked_dir, f"labels_mask_part_{index}.npy"), dtype=np.uint8)
        part["mask_not_binary"] = int((src > 1).sum() + (out > 1).sum())
        src, out = src != 0, out != 0
        t["out_trained_tokens"] += int(out.sum())
        final = ends - 1
        cleared = src & ~out
        cleared[final] = False
        part.update(
            {
                "inventory_docs": len(records),
                "inventory_tokens": int((ends - starts).sum()),
                "inventory_starting_im_start": int((tokens[starts] == im_start).sum()),
                "inventory_over_max_seq_length": int((ends - starts > max_seq_length).sum()),
                "dropped_at_final": int((src[final] & ~out[final]).sum()),
                "already_untrained_at_final": int((~src[final] & ~out[final]).sum()),
                "final_still_trained": int(out[final].sum()),
                "cleared_elsewhere": int(cleared.sum()),
                "labels_added": int((~src & out).sum()),
            }
        )
        t.update(part)
        per_part.append({"part": index, **part})

    for chat_dir in chat_dirs:
        found, orphans = list_parts(chat_dir)
        # Same part-set rule as the replay dirs: every part complete, no orphans, and at least one part checked.
        t["orphan_part_files"] += len(orphans)
        t["part_set_mismatch"] += int(
            not found["tokens"] or not (found["tokens"] == found["mask"] == found["metadata"])
        )
        for index in sorted(found["tokens"] & found["mask"] & found["metadata"]):
            num_tokens = os.path.getsize(os.path.join(chat_dir, f"labels_mask_part_{index}.npy"))
            rows, malformed = csv_rows(chat_dir, index)
            t["metadata_malformed_lines"] += malformed
            if not tiles(rows, num_tokens):
                t["untiled_parts"] += 1
                continue
            rows = rows[rows[:, 1] > rows[:, 0]]
            t["chat_docs"] += len(rows)
            t["chat_not_im_start"] += int((token_memmap(chat_dir, index, num_tokens)[rows[:, 0]] != im_start).sum())

    # OUT's own statistics, as main's startup check reads them (_check_row_aligned_cache): the row-aligned marker,
    # one row per metadata line, and loss tokens counted from OUT's masks.
    stats_path = os.path.join(masked_dir, "dataset_statistics.json")
    t["statistics_missing"] = int(not os.path.isfile(stats_path))
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        metadata_lines = 0
        for name in os.listdir(masked_dir):
            if name.startswith("token_ids_part_") and name.endswith(".csv.gz"):
                with gzip.open(os.path.join(masked_dir, name), "rt") as f:
                    metadata_lines += sum(1 for _ in f)
        overall = stats.get("overall_statistics", {})
        t["statistics_mismatch"] = int(
            stats.get("configuration", {}).get("row_aligned_parts") is not True
            or overall.get("total_instances") != metadata_lines
            or overall.get("trainable_tokens") != t["out_trained_tokens"]
        )

    ok = (
        all(t[k] == 0 for k in ZERO)
        and t["found"] == pins["rows"]
        and t["dropped_at_final"] + t["already_untrained_at_final"] == pins["rows"]
    )
    return {
        "pass": bool(ok),
        "totals": {k: int(v) for k, v in t.items()},
        "zero_checks": list(ZERO),
        "pins": pins,
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
    parser.add_argument(
        "--replay_dir", required=True, help="The source replay dir (unmasked), with its manifest.jsonl."
    )
    parser.add_argument("--masked_dir", required=True, help="mask_replay_final_eos.py's output for it.")
    parser.add_argument("--pins", choices=sorted(PINS), default="h182", help="The inventory pins to require.")
    parser.add_argument("--pins_json", help="Pins for another replay dir: manifest_sha256, ids_sha256, rows, tokens.")
    parser.add_argument("--chat_dir", action="append", default=[], help="Chat cache trained beside it; repeatable.")
    parser.add_argument("--max_seq_length", type=int, default=65536)
    parser.add_argument("--eos_token_id", type=int, default=EOS)
    parser.add_argument("--im_start_token_id", type=int, default=IM_START)
    parser.add_argument("--out", help="Write the full result, with per-part counts, here.")
    args = parser.parse_args(argv)
    if args.pins_json:
        with open(args.pins_json) as f:
            pins = json.load(f)
    else:
        pins = PINS[args.pins]
    result = check(
        args.replay_dir,
        args.masked_dir,
        pins,
        args.chat_dir,
        args.max_seq_length,
        args.eos_token_id,
        args.im_start_token_id,
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "per_part"}, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
