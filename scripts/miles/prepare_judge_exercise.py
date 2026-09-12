"""Prepare a tiny immutable four-domain Dolci exercise from the audited baseline manifest."""

import argparse
import asyncio
import collections
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from open_instruct.miles import code_rewards, judge_server, run_data, workflow
from open_instruct.miles.run_spec import RunSpec

MANIFEST = Path(
    "/weka/oe-adapt-default/robertb/olmo-miles/trial-data/dolci-think-20260908/tokenized-v2/rl-manifest.json"
)
CODE_URL = "https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program"


async def code_canaries():
    args = SimpleNamespace(code_api_url=CODE_URL, code_pass_rate_reward_threshold=0.99)
    results = []
    for program, tests, stdio, expected in (
        ("def add(a,b): return a+b", ["assert add(1,2)==3"], False, 1),
        ("def add(a,b): return 0", ["assert add(1,2)==3"], False, 0),
        ('print("2")', [{"input": "", "output": "2\n"}], True, 1),
        ('print("3")', [{"input": "", "output": "2\n"}], True, 0),
    ):
        score = await code_rewards.code_score(args, program, tests, stdio=stdio)
        if score != expected:
            raise RuntimeError("Code service known-answer canary failed")
        results.append(score)
    return results


def prepare(spec):
    output = Path(spec.data["prompt_data"]).parent
    if output.exists():
        raise ValueError("Preparation output already exists; do not overwrite immutable exercise inputs")
    tokenizer = run_data._tokenizer(Path(spec.model["source"]))
    inputs = {}
    partitions, provenance, _ = run_data._adopt(
        {"rl_manifest": str(MANIFEST)}, tokenizer, tokenizer.chat_template, inputs
    )
    quotas = {"math": 2, "ifeval": 2, "code": 1, "code_stdio": 1, "general-quality": 1, "general-quality_ref": 1}
    selected = {}
    for split, rows in partitions.items():
        buckets = collections.defaultdict(list)
        seen = set()
        for row in rows:
            name = row["metadata"]["verifiers"][0]["name"]
            if (
                name in quotas
                and row["input"] not in seen
                and len(buckets[name]) < (quotas[name] * 2 if split == "train" else 1)
            ):
                buckets[name].append(row)
                seen.add(row["input"])
        # The historical held-out manifest need not contain both code subtypes;
        # training must cover all types, held-out covers all available types.
        if split == "train" and any(len(buckets[name]) != count * 2 for name, count in quotas.items()):
            raise ValueError(f"Insufficient exercise coverage: { {k: len(v) for k, v in buckets.items()} }")
        selected[split] = []
        for step in range(2 if split == "train" else 1):
            for name, count in quotas.items():
                selected[split].extend(
                    buckets[name][step * count : (step + 1) * count] if split == "train" else buckets[name]
                )
    registry = {name: {"factory": factory} for name, factory in run_data.FACTORIES.items()}
    for name in ("code", "code_stdio"):
        registry[name] = {
            "factory": "open_instruct.miles.code_rewards.CodeVerifier",
            "config": {"api_url": CODE_URL, "stdio": name == "code_stdio"},
        }
    for name in spec.judges["judging"]["bindings"]:
        registry[name] = {"factory": "open_instruct.miles.judge_registry.NamedJudgeVerifier", "config": {"name": name}}
    for rows in selected.values():
        for row in rows:
            run_data._verify_row(row, tokenizer, 2048, registry)
    if {r["input"] for r in selected["train"]} & {r["input"] for r in selected["eval"]}:
        raise ValueError("Training/held-out overlap")
    # Read-only validation of already cached judge weights and template.
    judge_server.command(spec.judges["judges"]["general"], 12345)
    canaries = asyncio.run(code_canaries())
    output.mkdir(parents=True)
    for split, rows in selected.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    workflow.write_json(output / "verifiers.json", registry)
    report = {
        "passed": True,
        "sources": inputs,
        "template_sha256": provenance["template_sha256"],
        "counts": {k: len(v) for k, v in selected.items()},
        "code_canaries": canaries,
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()},
    }
    workflow.write_json(output / "preparation.json", report)
    workflow.write_json(Path("/output/preparation.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    prepare(RunSpec.load(parser.parse_args().config))
