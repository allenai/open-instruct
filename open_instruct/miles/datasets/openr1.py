"""Prepare reproducible OpenR1 problem partitions and teacher-only reference metadata."""

import argparse
import hashlib
import heapq
import importlib
import json
import re
from collections import Counter
from pathlib import Path

from open_instruct.miles.errors import InputError

DATASET = "open-r1/OpenR1-Math-220k"


def sections(solution, limit=10):
    """Partition reference prose while retaining an unexposed final section."""
    text = solution.strip()
    text = re.sub(r"^<think>\s*", "", text).replace("</think>", "")
    parts = re.split(r"\n\s*\n", text)
    parts = [part.strip() for part in parts if part.strip()]
    if len(parts) < 2:
        return []
    width = max(1, (len(parts) + limit - 1) // limit)
    return ["\n\n".join(parts[i : i + width]) for i in range(0, len(parts), width)]


def convert(row, revision, config):
    """Return a question-only RL row with a mechanically verified reference."""
    problem, answer, identity = (row.get(key) for key in ("problem", "answer", "uuid"))
    if not all(isinstance(value, str) and value.strip() for value in (problem, answer, identity)):
        return None, "missing_fields"
    if row.get("question_type") == "proof" or answer.strip().lower() == "proof":
        return None, "proof"
    if re.search(r"!\[|<img|\[asy\]|\b(?:figure|diagram|shown below)\b", problem, re.I):
        return None, "visual"
    generations = row.get("generations") or []
    correct = row.get("correctness_math_verify") or []
    complete = row.get("is_reasoning_complete") or []
    finishes = row.get("finish_reasons")
    candidates = []
    for i, text in enumerate(generations):
        if i >= len(correct) or correct[i] is not True or i >= len(complete) or complete[i] is not True:
            continue
        if finishes is not None and (i >= len(finishes) or finishes[i] != "stop"):
            continue
        if not isinstance(text, str):
            continue
        chunks = sections(text)
        if len(chunks) >= 2 and "\\boxed" not in chunks[0]:
            candidates.append((len(text), i, chunks))
    if not candidates:
        return None, "no_usable_verified_reference"
    _, index, chunks = min(candidates)
    question = problem.strip() + "\n\nPut your final answer in \\boxed{}."
    return {
        "messages": [{"role": "user", "content": question}],
        "ground_truth": answer.strip(),
        "metadata": {
            "prepared_sample_id": f"openr1:{identity}",
            "source_dataset": DATASET,
            "source_revision": revision,
            "source_config": config,
            "source_uuid": identity,
            "source": row.get("source"),
            "problem_type": row.get("problem_type"),
            "query": question,
            "verifiers": [{"name": "math", "target": answer.strip(), "weight": 1.0}],
            "bread": {"sections": chunks, "reference_index": index},
        },
    }, None


def prepare(rows, output, *, revision, config="default", train_count=1024, eval_count=128, seed=17):
    """Write disjoint samples with stable training identities and bounded memory."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise InputError("OpenR1 preparation requires an immutable 40-character HF revision")
    if config not in ("default", "extended", "all"):
        raise InputError("Unknown OpenR1 configuration")
    if train_count < 1 or eval_count < 1:
        raise InputError("Both train_count and eval_count must be positive")
    output = Path(output)
    if output.exists():
        raise InputError("Output already exists; use a fresh preparation directory")
    count = train_count + eval_count
    reservoir, seen = [], set()
    stats = Counter()
    for raw in rows:
        row, reason = convert(raw, revision, config)
        stats["scanned"] += 1
        if row is None:
            stats[reason] += 1
            continue
        # Collapse repeated statements before assigning either partition.
        key = hashlib.sha256(" ".join(raw["problem"].split()).encode()).hexdigest()
        identity = raw["uuid"]
        if key in seen or identity in seen:
            stats["duplicate"] += 1
            continue
        seen.update((key, identity))
        stats["eligible"] += 1
        priority = int(hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest(), 16)
        entry = (-priority, identity, row)
        if len(reservoir) < count:
            heapq.heappush(reservoir, entry)
        elif entry > reservoir[0]:
            heapq.heapreplace(reservoir, entry)
    if len(reservoir) < count:
        raise InputError(f"Requested {count} examples but only {len(reservoir)} eligible unique problems remain")
    selected = [entry[2] for entry in sorted(reservoir, reverse=True)]
    output.mkdir(parents=True)
    artifacts = {}
    for name, values in (("train", selected[:train_count]), ("eval", selected[train_count:])):
        if name == "eval":
            for row in values:
                del row["metadata"]["bread"]
        raw = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in values).encode()
        (output / f"{name}.jsonl").write_bytes(raw)
        artifacts[name] = {"path": f"{name}.jsonl", "sha256": hashlib.sha256(raw).hexdigest(), "records": len(values)}
    manifest = {
        "schema_version": 1,
        "source": {"dataset": DATASET, "revision": revision, "config": config, "license": "apache-2.0"},
        "seed": seed,
        "statistics": dict(stats),
        "miles": {
            "input_key": "messages",
            "label_key": "ground_truth",
            "metadata_key": "metadata",
            "apply_chat_template": True,
            "custom_rm_path": "olmo_miles.rl.rewards.registered_reward",
        },
        "artifacts": artifacts,
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Immutable Hugging Face dataset commit")
    parser.add_argument("--config", choices=("default", "extended", "all"), default="default")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--eval-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be an immutable 40-character HF commit")
    rows = importlib.import_module("datasets").load_dataset(
        DATASET, args.config, revision=args.revision, split="train", streaming=True
    )
    print(
        prepare(
            rows,
            args.output,
            revision=args.revision,
            config=args.config,
            train_count=args.train_count,
            eval_count=args.eval_count,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
