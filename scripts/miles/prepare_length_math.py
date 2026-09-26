"""Select existing held-out-disjoint math rows for response-length qualification."""

import argparse
import hashlib
import json
from pathlib import Path

from open_instruct.miles.datasets import run_data


def prepare(source, output, model):
    if output.exists():
        raise ValueError("Choose a new immutable output directory")
    tokenizer = run_data._tokenizer(model)
    report = {"source": str(source), "model": str(model), "splits": {}}
    selected = {}
    seen = set()
    for split in ("train", "eval"):
        selected[split] = []
        for line in (source / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            if [v["name"] for v in row["metadata"]["verifiers"]] != ["math"]:
                continue
            tokens = tokenizer.encode(row["input"], add_special_tokens=False)
            if len(tokens) > 2048:
                raise ValueError("Math prompt exceeds reserved input budget")
            key = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
            if key in seen:
                raise ValueError("Duplicate or overlapping prompts")
            seen.add(key)
            selected[split].append(row)
        if len(selected[split]) < (8 if split == "train" else 2):
            raise ValueError(f"Insufficient math rows in {split}")
        report["splits"][split] = [
            {
                "id": r["metadata"]["prepared_sample_id"],
                "prompt_tokens": len(tokenizer.encode(r["input"], add_special_tokens=False)),
            }
            for r in selected[split]
        ]
    output.mkdir(parents=True)
    for split, rows in selected.items():
        data = "".join(json.dumps(row) + "\n" for row in rows)
        (output / f"{split}.jsonl").write_text(data)
        report[split + "_sha256"] = hashlib.sha256(data.encode()).hexdigest()
    registry = json.loads((source / "verifiers.json").read_text())
    (output / "verifiers.json").write_text(json.dumps({"math": registry["math"]}, indent=2) + "\n")
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("/output/length-math-preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output, args.model)
