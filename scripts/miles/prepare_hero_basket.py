"""Render the frozen mixed basket with a hero checkpoint's own chat template."""

import argparse
import asyncio
import collections
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from scripts.miles import prepare_baseline_basket, prepare_olmo3_basket

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.datasets import run_data
from open_instruct.miles.execution import workflow

BASELINE_PROVENANCE = (
    Path(__file__).resolve().parents[2] / "docs/miles/measurements/full-sft-basket-20260914/preparation.json"
)


def prompt_map(tokenizer, inputs):
    path = prepare_baseline_basket.MANIFEST
    manifest = run_data._read_json_object(path, inputs)
    options = manifest["miles"]
    descriptor = options.get("chat_template")
    original_tokenizer = tokenizer
    if descriptor is None:
        # The original adoption workflow used its policy tokenizer when the
        # manifest omitted a template. Reconstruct that historical rendering,
        # not a rendering under the new hero's template or special tokens.
        baseline = run_data._read_json_object(BASELINE_PROVENANCE, inputs)
        source = Path(baseline["model"]["source"])
        original_tokenizer = run_data._tokenizer(source)
        template = original_tokenizer.chat_template.encode()
        expected = baseline["template_sha256"]
        for name in ("tokenizer_config.json", "chat_template.jinja"):
            if (source / name).is_file():
                run_data._read(source / name, inputs)
    else:
        template = run_data._read(path.parent / descriptor["path"], inputs)
        expected = descriptor["sha256"]
    if run_data._sha(template) != expected:
        raise ValueError("Original chat template hash mismatch")
    mapping = {}
    for artifact in manifest["artifacts"].values():
        raw = run_data._read(path.parent / artifact["path"], inputs)
        if run_data._sha(raw) != artifact["sha256"]:
            raise ValueError("Canonical source artifact hash mismatch")
        rows = run_data._rows(raw)
        if len(rows) != artifact["records"]:
            raise ValueError("Canonical source count mismatch")
        for row in rows:
            messages = run_data._messages({"messages": row[options["input_key"]]}, strip_answer=False)
            old = run_data._render(messages, original_tokenizer, template.decode())
            new = run_data._render(messages, tokenizer, tokenizer.chat_template)
            if old in mapping and mapping[old] != new:
                raise ValueError("Ambiguous canonical messages for frozen prompt")
            mapping[old] = new
    return mapping


def prepare(spec):
    output = Path(spec.data["prompt_data"]).parent
    if output.exists():
        raise ValueError(f"Immutable preparation already exists: {output}")
    source = prepare_olmo3_basket.SOURCE
    inputs = {}
    for name, expected in prepare_olmo3_basket.SOURCE_HASHES.items():
        raw = run_data._read(source / name, inputs)
        if run_data._sha(raw) != expected:
            raise ValueError(f"Frozen basket hash mismatch: {name}")
    tokenizer = run_data._tokenizer(Path(spec.model["source"]))
    mapping = prompt_map(tokenizer, inputs)
    registry = json.loads((source / "verifiers.json").read_text())
    limit = spec.compile().miles["rollout_max_prompt_len"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    counts, excluded, changed = collections.Counter(), collections.Counter(), collections.Counter()
    domains, ids, prompts = {}, {}, {}
    try:
        shutil.copyfile(source / "verifiers.json", temporary / "verifiers.json")
        for path in sorted(source.glob("*.jsonl")):
            if path.name not in prepare_olmo3_basket.SOURCE_HASHES:
                raise ValueError(f"Unpinned basket file: {path}")
            split = path.stem
            domains[split], ids[split], prompts[split] = collections.Counter(), set(), set()
            with path.open() as reader, (temporary / path.name).open("w") as writer:
                for line in reader:
                    row = json.loads(line)
                    old = row["input"]
                    if old not in mapping:
                        raise ValueError("Frozen prompt missing from canonical messages")
                    row["input"] = mapping[old]
                    metadata = row["metadata"]
                    if "prompt_token_ids" in metadata:
                        metadata["source_prompt_token_ids"] = metadata.pop("prompt_token_ids")
                    if len(tokenizer.encode(row["input"], add_special_tokens=False)) > limit:
                        excluded[f"{split}/{prepare_baseline_basket.domain(row)}"] += 1
                        continue
                    run_data._verify_row(row, tokenizer, limit, registry)
                    ids[split].add(metadata["prepared_sample_id"])
                    prompts[split].add(row["input"])
                    domains[split][prepare_baseline_basket.domain(row)] += 1
                    counts[split] += 1
                    changed[split] += row["input"] != old
                    writer.write(json.dumps(row) + "\n")
        for split in ids:
            if split != "train" and (ids["train"] & ids[split] or prompts["train"] & prompts[split]):
                raise ValueError(f"Training/holdout overlap after rendering: {split}")
        if not counts["train"] or set(domains["train"]) != set(prepare_baseline_basket.DOMAINS.values()):
            raise ValueError("Prepared training data must retain all four domains")
        canaries = asyncio.run(prepare_baseline_basket.code_canaries())
        report = dict(
            passed=True,
            model=spec.model,
            sources=inputs,
            template_sha256=run_data._sha(tokenizer.chat_template.encode()),
            counts=dict(counts),
            domain_counts={k: dict(v) for k, v in domains.items()},
            excluded_overlong=dict(excluded),
            changed_renderings=dict(changed),
            max_prompt_tokens=limit,
            code_canaries=canaries,
            held_out_ids={k: sorted(v) for k, v in ids.items() if k != "train"},
            outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in temporary.iterdir()},
        )
        workflow.write_json(temporary / "preparation.json", report)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    workflow.write_json("/output/preparation.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "held_out_ids"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    prepare(RunSpec.load(args.config))
