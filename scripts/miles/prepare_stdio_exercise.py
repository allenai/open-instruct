"""Select short natural stdio problems with original labels and split identities."""

import argparse
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.miles import prepare_colleague_exercises as preparation


def prepare(model, output):
    manifest, partitions, inputs = preparation.source_rows()
    options = manifest["miles"]
    selected = {}
    for split, sources in partitions.items():
        rows = []
        for index, source in enumerate(sources):
            metadata = source[options["metadata_key"]]
            if metadata["verifiers"][0]["name"] != "code_stdio":
                continue
            source = copy.deepcopy(source)
            source[options["metadata_key"]].setdefault("prepared_sample_id", f"manifest:{split}:{index}")
            rows.append(source)
        selected[split] = sorted(rows, key=lambda r: len(json.dumps(r[options["input_key"]])))
    spec = SimpleNamespace(model={"source": model}, data={"prompt_data": str(Path(output) / "train.jsonl")}, judges={})
    canaries = asyncio.run(preparation.canaries())
    report = preparation.select(spec, manifest, selected, {"code_stdio": 4}, 2, eval_per_domain=2)
    report = {
        **report,
        "sources": inputs,
        "canaries": canaries,
        "selection": "Shortest original rendered-message JSON by character count within each split; not a difficulty estimate or reward-based selection.",
    }
    report["queries"] = {
        split: [
            json.loads(line)["metadata"]["query"]
            for line in (Path(output) / f"{split}.jsonl").read_text().splitlines()
        ]
        for split in ("train", "eval")
    }
    Path("/output/stdio-preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("output")
    args = parser.parse_args()
    prepare(args.model, args.output)


if __name__ == "__main__":
    main()
