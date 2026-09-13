"""Select short natural stdio problems with original labels and split identities."""

import argparse
import asyncio
import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

from scripts.miles import prepare_colleague_exercises as preparation


def has_statement(prompt):
    """Conservative qualification-only screen; selected prompts still need manual review."""
    # This manifest includes entries where scraping retained only examples.
    body = prompt.split("where CODE is the solution for the problem.", 1)[-1]
    body = body.split("Write Python code to solve the problem.", 1)[0]
    body = re.sub(r"(?im)^.*(?:time limit|memory limit).*$", "", body)
    if not all(re.search(rf"(?im)^\s*(?:[#-]+\s*)?{name}\b", body) for name in ("input", "output")):
        return False
    statement = re.split(
        r"(?im)^\s*(?:[#-]+\s*)?(?:examples?|sample(?: input| output)?|input|output)\b", body, maxsplit=1
    )[0]
    return (
        bool(re.search(r"(?i)\b(?:compute|calculate|given|find|determine|your task|you have|there are)\b", statement))
        and sum(character.isalpha() for character in statement) >= 8
    )


def prepare(model, output):
    manifest, partitions, inputs = preparation.source_rows()
    options = manifest["miles"]
    selected = {}
    rejected = {}
    for split, sources in partitions.items():
        rows = []
        rejected[split] = 0
        for index, source in enumerate(sources):
            metadata = source[options["metadata_key"]]
            if metadata["verifiers"][0]["name"] != "code_stdio":
                continue
            messages = preparation.run_data._messages({"messages": source[options["input_key"]]}, strip_answer=False)
            if not has_statement(messages[-1]["content"]):
                rejected[split] += 1
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
        "missing_statement_rejections": rejected,
        "selection": "Shortest original rendered-message JSON with a statement and explicit input/output headings within each split; not a difficulty estimate or reward-based selection.",
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
