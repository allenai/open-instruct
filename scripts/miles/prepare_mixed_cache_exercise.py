"""Prepare an immutable math/IF/code Dolci mixture for the radix-cache A/B runs.

Adopts the audited baseline manifest like the judge exercise, but selects only the
verifier-scored domains (no judge services): ``math``, ``ifeval``, ``code`` and
``code_stdio``, with equal per-domain quotas. Code prompts are the longest in the
mixture, which is where prefix reuse across a prompt's samples has the most to
cache. Writes ``train.jsonl``, ``eval.jsonl``, ``verifiers.json`` and a
``preparation.json`` report next to the run file's ``data.prompt_data`` path.
"""

import argparse
import asyncio
import collections
import hashlib
import json
from pathlib import Path

from scripts.miles.prepare_judge_exercise import CODE_URL, MANIFEST, code_canaries

from open_instruct.miles import run_data, workflow
from open_instruct.miles.run_spec import RunSpec

DOMAINS = ("math", "ifeval", "code", "code_stdio")


def prepare(spec, *, train_per_domain, eval_per_domain):
    output = Path(spec.data["prompt_data"]).parent
    if output.exists():
        raise ValueError("Preparation output already exists; do not overwrite immutable exercise inputs")
    tokenizer = run_data._tokenizer(Path(spec.model["source"]))
    inputs = {}
    partitions, provenance, _ = run_data._adopt(
        {"rl_manifest": str(MANIFEST)}, tokenizer, tokenizer.chat_template, inputs
    )
    quotas = {"train": train_per_domain, "eval": eval_per_domain}
    selected = {}
    seen = set()
    for split in ("train", "eval"):
        buckets = collections.defaultdict(list)
        for row in partitions[split]:
            name = row["metadata"]["verifiers"][0]["name"]
            if name in DOMAINS and row["input"] not in seen and len(buckets[name]) < quotas[split]:
                buckets[name].append(row)
                seen.add(row["input"])
        short = {name: len(buckets[name]) for name in DOMAINS if len(buckets[name]) < quotas[split]}
        if short:
            raise ValueError(f"Insufficient {split} coverage: {short}")
        # Interleave domains so every collection of consecutive prompts is mixed.
        selected[split] = [row for group in zip(*(buckets[name] for name in DOMAINS), strict=True) for row in group]
    registry = {name: {"factory": factory} for name, factory in run_data.FACTORIES.items()}
    for name in ("code", "code_stdio"):
        registry[name] = {
            "factory": "open_instruct.miles.code_rewards.CodeVerifier",
            "config": {"api_url": CODE_URL, "stdio": name == "code_stdio"},
        }
    for rows in selected.values():
        for row in rows:
            run_data._verify_row(row, tokenizer, 2048, registry)
    if {r["input"] for r in selected["train"]} & {r["input"] for r in selected["eval"]}:
        raise ValueError("Training/held-out overlap")
    canaries = asyncio.run(code_canaries())
    output.mkdir(parents=True)
    for split, rows in selected.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    workflow.write_json(output / "verifiers.json", registry)
    lengths = collections.defaultdict(list)
    for row in selected["train"]:
        lengths[row["metadata"]["verifiers"][0]["name"]].append(row["metadata"]["run_prompt_tokens"])
    report = {
        "passed": True,
        "sources": inputs,
        "template_sha256": provenance["template_sha256"],
        "counts": {k: len(v) for k, v in selected.items()},
        "train_prompt_tokens_by_domain": {
            name: {"mean": sum(v) / len(v), "max": max(v), "count": len(v)} for name, v in lengths.items()
        },
        "code_canaries": canaries,
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()},
    }
    workflow.write_json(output / "preparation.json", report)
    workflow.write_json(Path("/output/preparation.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--train-per-domain", type=int, default=320)
    parser.add_argument("--eval-per-domain", type=int, default=32)
    options = parser.parse_args()
    prepare(
        RunSpec.load(options.config),
        train_per_domain=options.train_per_domain,
        eval_per_domain=options.eval_per_domain,
    )
