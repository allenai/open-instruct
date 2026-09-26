"""Freeze a four-domain baseline, preserving historical held-out prompts and source weights."""

import argparse
import asyncio
import collections
import hashlib
import json
import time
from pathlib import Path

from scripts.miles.prepare_judge_exercise import CODE_URL, MANIFEST, code_canaries

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.datasets import run_data
from open_instruct.miles.execution import workflow
from open_instruct.miles.rewards import judge_server

DOMAINS = {
    "math": "math",
    "ifeval": "ifeval",
    "code": "code",
    "code_stdio": "code",
    "general-quality": "general",
    "general-quality_ref": "general",
}


def domain(row):
    return DOMAINS[row["metadata"]["verifiers"][0]["name"]]


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def select(partitions, *, count=128, seed=17):
    """Grow fixed holdouts; remove every matching prompt/source identity from training."""
    if type(count) is not int or count < 1:
        raise ValueError("Held-out count must be a positive integer")
    selected = collections.defaultdict(list)
    prompts, identities = set(), set()

    def add(row):
        key = domain(row)
        identity = row["metadata"].get("prepared_sample_id")
        if row["input"] in prompts or (identity and identity in identities):
            return
        selected[key].append(row)
        prompts.add(row["input"])
        if identity:
            identities.add(identity)

    for row in partitions["eval"]:
        add(row)
    if any(len(rows) > count for rows in selected.values()):
        raise ValueError("Requested holdout would discard historical held-out identities")
    # Hash order is stable across source order changes and independent of model outputs.
    for row in sorted(partitions["train"], key=lambda row: digest(f"{seed}:{row['input']}")):
        if len(selected[domain(row)]) < count:
            add(row)
    if any(len(selected[name]) != count for name in set(DOMAINS.values())):
        raise ValueError(f"Insufficient distinct held-out prompts: { {k: len(v) for k, v in selected.items()} }")
    train = [
        row
        for row in partitions["train"]
        if row["input"] not in prompts and row["metadata"].get("prepared_sample_id") not in identities
    ]
    if not train:
        raise ValueError("No training rows remain after holding out evaluation")
    return {"train": train, **{f"eval-{name}": selected[name] for name in sorted(selected)}}


def prepare(spec):
    output = Path(spec.data["prompt_data"]).parent
    if output.exists():
        raise ValueError(f"Immutable preparation output already exists: {output}; use a new baseline identity")
    tokenizer = run_data._tokenizer(Path(spec.model["source"]))
    inputs = {}
    partitions, provenance, _ = run_data._adopt(
        {"rl_manifest": str(MANIFEST)}, tokenizer, tokenizer.chat_template, inputs
    )
    if provenance["template_sha256"] != digest(tokenizer.chat_template):
        raise ValueError("Baseline manifest and policy checkpoint chat templates differ")
    registry = {name: {"factory": factory} for name, factory in run_data.FACTORIES.items()}
    for name in ("code", "code_stdio"):
        registry[name] = {
            "factory": "open_instruct.miles.rewards.code_rewards.CodeVerifier",
            "config": {"api_url": CODE_URL, "stdio": name == "code_stdio"},
        }
    for name in spec.judges["judging"]["bindings"]:
        registry[name] = {"factory": "open_instruct.miles.rewards.judge_registry.NamedJudgeVerifier", "config": {"name": name}}
    limit = spec.compile().miles["rollout_max_prompt_len"]
    excluded = collections.Counter()
    eligible = {}
    for split, rows in partitions.items():
        eligible[split] = []
        for row in rows:
            key = domain(row)  # Unknown verifier/domain is a hard error.
            length = len(tokenizer.encode(row["input"], add_special_tokens=False))
            if length > limit:
                excluded[f"{split}/{key}/overlong"] += 1
                continue
            run_data._verify_row(row, tokenizer, limit, registry)
            eligible[split].append(row)
    selected = select(eligible)
    # Preserve original source weighting/duplicates in training. Only holdout removal
    # and the explicit prompt budget filter alter source proportions.
    for rows in selected.values():
        for row in rows:
            row["metadata"].setdefault("prepared_sample_id", "basket:" + digest(row["input"]))
    train_ids = {row["metadata"]["prepared_sample_id"] for row in selected["train"]}
    eval_ids = {row["metadata"]["prepared_sample_id"] for k, rows in selected.items() if k != "train" for row in rows}
    if train_ids & eval_ids:
        raise ValueError("Training/held-out source identity overlap")
    judge_server.command(spec.judges["judges"]["general"], 12345)
    canaries = asyncio.run(code_canaries())
    output.mkdir(parents=True)
    for split, rows in selected.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    workflow.write_json(output / "verifiers.json", registry)
    report = {
        "passed": True,
        "source_manifest": str(MANIFEST),
        "sources": inputs,
        "model": spec.model,
        "template_sha256": provenance["template_sha256"],
        "seed": 17,
        "max_prompt_tokens": limit,
        "excluded": dict(excluded),
        "counts": {key: len(rows) for key, rows in selected.items()},
        "verifier_counts": {
            key: dict(collections.Counter(row["metadata"]["verifiers"][0]["name"] for row in rows))
            for key, rows in selected.items()
        },
        "held_out_ids": {
            key: [r["metadata"]["prepared_sample_id"] for r in rows]
            for key, rows in selected.items()
            if key != "train"
        },
        "code_canaries": canaries,
        "outputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()},
    }
    workflow.write_json(output / "preparation.json", report)
    workflow.write_json(Path("/output/preparation.json"), report)
    print(json.dumps({key: value for key, value in report.items() if key != "held_out_ids"}, indent=2), flush=True)


def verify(spec, *, wait_seconds=0):
    output = Path(spec.data["prompt_data"]).parent
    manifest = output / "preparation.json"
    deadline = time.monotonic() + wait_seconds
    while not manifest.is_file() and time.monotonic() < deadline:
        time.sleep(5)
    report = json.loads(manifest.read_text())
    if not report["passed"] or report["model"] != spec.model:
        raise ValueError("Preparation checkpoint identity differs from the submitted baseline")
    for name, expected in report["outputs"].items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen baseline artifact changed: {name}")
    for source, expected in report["sources"].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Baseline source changed: {source}")
    workflow.write_json(Path("/output/preparation.json"), report)
    print("Frozen baseline preparation and source hashes verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    spec = RunSpec.load(args.config)
    if args.verify:
        verify(spec, wait_seconds=args.wait_seconds)
    else:
        prepare(spec)
