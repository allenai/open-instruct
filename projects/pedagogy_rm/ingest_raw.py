"""Turn a rater's loose JSONL into the labels file the rest of the pipeline reads.

    python projects/pedagogy_rm/ingest_raw.py \
        --raw-dir data/eval50/labels/raw \
        --units data/eval50/pool.json \
        --shots data/eval50/rater_pack/shots.json \
        --out-dir data/eval50/labels

WHY A SEPARATE INGEST RATHER THAN HAVING RATERS WRITE THE REAL FORMAT. label_agents.py
talks to the gateway and controls the reply, so it can build a well-formed file directly.
A rater that is itself an agent working in the repo cannot be trusted to hand-assemble a
180-element JSON array with a `shots` block: one truncated write and the whole file is
unparseable, and the failure arrives after the expensive part is already done.

JSONL is the format that fails gracefully. A line is a complete record, so a run that dies
at 140 turns leaves 140 usable ones, and appending is safe. This script is what puts the
envelope back on - the schema tag, the rater name, and the `shots` block that agreement.py
needs in order to refuse to score a rater on turns it was shown the answer to.

VALIDATION HAPPENS HERE AND IS LOUD. A rater that emitted a string, a null, a 4, or the
same score for every turn produced a file that looks fine and means nothing. Those are
reported per rater rather than silently dropped, because which rater degraded and how is
itself a result: it is the difference between "the dimension is hard" and "one model
stopped trying".
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics

from projects.pedagogy_rm.rubric import BY_KEY


def read_jsonl(path: str) -> tuple[list[dict], list[str]]:
    """Records and complaints. A bad line is skipped and named, never guessed at."""
    records, problems = [], []
    with open(path) as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip().strip(",")
            if not line or line in "[]":
                continue
            try:
                blob = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"line {number} is not JSON")
                continue
            if isinstance(blob, list):  # a rater that wrapped a batch in an array
                records.extend(b for b in blob if isinstance(b, dict))
            elif isinstance(blob, dict):
                records.append(blob)
            else:
                problems.append(f"line {number} is a {type(blob).__name__}")
    return records, problems


def clean(records: list[dict], keys: list[str], known: set[str]) -> tuple[dict[str, dict], list[str]]:
    """The last record per id, keeping only well-formed scores."""
    kept: dict[str, dict] = {}
    problems: list[str] = []
    for record in records:
        uid = record.get("id")
        if uid not in known:
            problems.append(f"unknown id {uid!r}")
            continue
        out = {"id": uid, "flag": str(record.get("flag") or "")}
        for key in keys:
            value = record.get(key)
            # Booleans are ints in Python and would pass the range check as 1; a rater that
            # answered true/false has not answered this rubric and should be caught, not cast.
            if isinstance(value, bool) or not isinstance(value, int):
                problems.append(f"{uid} {key}={value!r}")
                continue
            if not BY_KEY[key].lo <= value <= BY_KEY[key].hi:
                problems.append(f"{uid} {key}={value} out of range")
                continue
            out[key] = value
        kept[uid] = out
    return kept, problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/eval50/labels/raw")
    parser.add_argument("--units", default="data/eval50/pool.json")
    parser.add_argument("--shots", default="data/eval50/rater_pack/shots.json")
    parser.add_argument("--out-dir", default="data/eval50/labels")
    parser.add_argument("--prefix", default="agent_", help="written as <prefix><rater>.json")
    args = parser.parse_args()

    with open(args.units) as handle:
        known = {u["id"] for u in json.load(handle)["units"]}
    with open(args.shots) as handle:
        blob = json.load(handle)
    keys, shot_ids = blob["dimensions"], sorted(blob["ids"])

    paths = sorted(glob.glob(os.path.join(args.raw_dir, "*.jsonl")))
    if not paths:
        raise SystemExit(f"nothing in {args.raw_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"{len(known)} units, {len(shot_ids)} of them shown to every rater as demonstrations\n")
    print(f"{'rater':<12} {'rated':>6} {'missing':>8} {'bad':>5}   per-dimension spread (sd)")
    print("-" * 78)
    counts: collections.Counter[str] = collections.Counter()
    for path in paths:
        name = os.path.basename(path).removesuffix(".jsonl")
        records, problems = read_jsonl(path)
        kept, more = clean(records, keys, known)
        problems += more
        counts.update(kept.keys())

        out = {
            "schema": "pedagogy-rm/labels-v1",
            "rater": name,
            "model": name,
            "source": path,
            "shots": {key: shot_ids for key in keys},
            "labels": list(kept.values()),
        }
        with open(os.path.join(args.out_dir, f"{args.prefix}{name}.json"), "w") as handle:
            json.dump(out, handle, indent=1)

        spread = []
        for key in keys:
            values = [r[key] for r in kept.values() if key in r]
            sd = statistics.pstdev(values) if len(values) > 1 else 0.0
            spread.append(f"{key[:4]} {sd:.2f}")
        flat = [key for key in keys if len({r.get(key) for r in kept.values()}) <= 1]
        print(f"{name:<12} {len(kept):>6} {len(known) - len(kept):>8} {len(problems):>5}   {'  '.join(spread)}")
        if flat:
            print(f"{'':<12} CONSTANT on {flat} — that rater is not discriminating, drop it")
        for line in problems[:3]:
            print(f"{'':<12} - {line}")
        if len(problems) > 3:
            print(f"{'':<12} - and {len(problems) - 3} more")

    print(
        f"\n{len(counts)}/{len(known)} units have at least one rating; "
        f"{sum(1 for c in counts.values() if c >= 3)} have three or more"
    )


if __name__ == "__main__":
    main()
