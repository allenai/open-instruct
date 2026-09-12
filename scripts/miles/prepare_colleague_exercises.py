"""Prepare immutable colleague fixtures and first-use canaries on CPU/WEKA (Saturn)."""

import argparse
import asyncio
import collections
import copy
import hashlib
import json
import multiprocessing
import os
import tempfile
import time
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

from scripts.miles.prepare_judge_exercise import CODE_URL, MANIFEST

from open_instruct.miles import code_rewards, judge_server, run_data, workflow
from open_instruct.miles.run_spec import RunSpec


def digest(value):
    return hashlib.sha256(value).hexdigest()


def import_probe(model):
    # Spawned workers race the same fresh remote-module cache, as Ray actors do.
    tokenizer = run_data._tokenizer(Path(model))
    config = json.loads((Path(model) / "config.json").read_text())
    return {
        "pid": os.getpid(),
        "model_type": config["model_type"],
        "template_sha256": digest(tokenizer.chat_template.encode()),
        "token_ids": tokenizer.encode("Concurrent first-use tokenizer canary", add_special_tokens=False),
    }


def inventory(model):
    model = Path(model)
    indices = list(model.glob("*.safetensors.index.json"))
    shards = (
        {name for index in indices for name in json.loads(index.read_text())["weight_map"].values()}
        if indices
        else {p.name for p in model.glob("*.safetensors")}
    )
    if not shards or any(not (model / name).is_file() for name in shards):
        raise ValueError(f"Incomplete checkpoint shard inventory: {model}")
    return {
        "path": str(model),
        "descriptor_sha256": digest((model / "config.json").read_bytes()),
        "shards": {name: (model / name).stat().st_size for name in sorted(shards)},
        "weight_values_checked": False,
    }


async def canaries():
    args = SimpleNamespace(code_api_url=CODE_URL, code_pass_rate_reward_threshold=0.99, code_max_execution_time=1.0)
    results = []
    for name, program, target, stdio, expected in (
        ("function-correct", "def add(a,b): return a+b", ["assert add(1,2)==3"], False, 1),
        ("function-wrong", "def add(a,b): return 0", ["assert add(1,2)==3"], False, 0),
        ("stdio-correct", 'print("2")', [{"input": "", "output": "2\n"}], True, 1),
        ("stdio-wrong", 'print("3")', [{"input": "", "output": "2\n"}], True, 0),
        ("syntax-error", "def broken(", ["assert True"], False, 0),
        ("execution-timeout", "while True: pass", [{"input": "", "output": "2\n"}], True, 0),
    ):
        score, detail = await code_rewards.execute(args, program, target, stdio=stdio)
        result = {"name": name, "score": score, "expected": expected, "diagnostics": detail}
        results.append(result)
        # A rejected/errored service response must not masquerade as a passing zero canary.
        if score != expected or detail["status"] != "ok":
            raise RuntimeError(f"Code canary did not establish execution semantics: {result}")
    return results


def source_rows():
    inputs = {}
    manifest = run_data._read_json_object(MANIFEST, inputs)
    partitions = {}
    for split, artifact in manifest["artifacts"].items():
        raw = run_data._read(MANIFEST.parent / artifact["path"], inputs)
        if digest(raw) != artifact["sha256"]:
            raise ValueError(f"Source artifact digest mismatch: {split}")
        rows = run_data._rows(raw)
        if len(rows) != artifact["records"]:
            raise ValueError(f"Source artifact row count mismatch: {split}")
        partitions[split] = rows
    return manifest, partitions, inputs


def select(spec, manifest, partitions, quotas, collections_count):
    tokenizer = run_data._tokenizer(Path(spec.model["source"]))
    options = manifest["miles"]
    registry = {name: {"factory": factory} for name, factory in run_data.FACTORIES.items()}
    for name in ("code", "code_stdio"):
        registry[name] = {
            "factory": "open_instruct.miles.code_rewards.CodeVerifier",
            "config": {"api_url": CODE_URL, "stdio": name == "code_stdio", "pass_rate_reward_threshold": 0.99},
        }
    for name in spec.judges.get("judging", {}).get("bindings", {}):
        registry[name] = {"factory": "open_instruct.miles.judge_registry.NamedJudgeVerifier", "config": {"name": name}}
    selected, seen_content, seen_ids = {}, set(), set()
    dropped = collections.Counter()
    # Reserve held-out identities first, then exclude them from training, across all domains.
    for split in ("eval", "train"):
        buckets = collections.defaultdict(list)
        multiplier = collections_count if split == "train" else 1
        for index, source in enumerate(partitions[split]):
            metadata = copy.deepcopy(source[options["metadata_key"]])
            name = metadata["verifiers"][0]["name"]
            count = quotas.get(name, 0) * multiplier if split == "train" else (2 if name in quotas else 0)
            if len(buckets[name]) >= count:
                continue
            messages = run_data._messages({"messages": source[options["input_key"]]}, strip_answer=False)
            normalized = json.dumps([(m["role"], " ".join(m["content"].split())) for m in messages])
            content_id = digest(normalized.encode())
            identity = metadata.get("prepared_sample_id", f"manifest:{split}:{index}")
            if content_id in seen_content or identity in seen_ids:
                dropped[f"{split}:duplicate"] += 1
                continue
            # The original manifest's token IDs belong to its tokenizer/template.
            # Retain their hash and explicitly render this model's pinned HF template.
            old_ids = metadata.pop("prompt_token_ids", None)
            if old_ids is not None:
                metadata["source_prompt_token_ids_sha256"] = digest(json.dumps(old_ids).encode())
            metadata.update(
                prepared_sample_id=identity, query=messages[-1]["content"], source_content_sha256=content_id
            )
            row = {
                "input": run_data._render(messages, tokenizer, tokenizer.chat_template),
                "label": source[options["label_key"]],
                "metadata": metadata,
            }
            ids = tokenizer.encode(row["input"], add_special_tokens=False)
            if not 0 < len(ids) <= 2048:
                dropped[f"{split}:prompt_budget"] += 1
                continue
            run_data._verify_row(row, tokenizer, 2048, registry)
            buckets[name].append(row)
            seen_content.add(content_id)
            seen_ids.add(identity)
        required = {name: (count * multiplier if split == "train" else 2) for name, count in quotas.items()}
        if any(len(buckets[name]) < count for name, count in required.items()):
            raise ValueError(
                f"Insufficient {split} coverage: required={required}, found={ {k: len(v) for k, v in buckets.items()} }"
            )
        selected[split] = []
        for batch in range(multiplier):
            for name, quota in quotas.items():
                count = quota if split == "train" else 2
                selected[split].extend(buckets[name][batch * count : (batch + 1) * count])
    output = Path(spec.data["prompt_data"]).parent
    output.mkdir(parents=True, exist_ok=False)
    for split, rows in selected.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    registry = {name: registry[name] for name in quotas}
    workflow.write_json(output / "verifiers.json", registry)
    report = {
        "counts": {
            split: dict(collections.Counter(row["metadata"]["verifiers"][0]["name"] for row in rows))
            for split, rows in selected.items()
        },
        "template_sha256": digest(tokenizer.chat_template.encode()),
        "dropped": dict(dropped),
        "normalized_content_and_identity_disjoint": True,
        "outputs": {p.name: digest(p.read_bytes()) for p in output.iterdir()},
    }
    workflow.write_json(output / "preparation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    moe = RunSpec.load(args.config)
    dense = RunSpec.load(Path("configs/miles/qualification/colleague-20260912/01-dense.toml"))
    report = {"passed": False, "started": time.time()}
    try:
        report["inventories"] = [inventory(spec.model["source"]) for spec in (dense, moe)]
        with tempfile.TemporaryDirectory(prefix="colleague-imports-") as cache:
            os.environ["HF_MODULES_CACHE"] = cache
            with futures.ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
                report["concurrent_imports"] = list(pool.map(import_probe, [moe.model["source"]] * 4))
            probes = report["concurrent_imports"]
            if (
                len({json.dumps(p["token_ids"]) for p in probes}) != 1
                or len({p["template_sha256"] for p in probes}) != 1
            ):
                raise ValueError("Concurrent tokenizer imports disagree")
        judge_server.command(moe.judges["judges"]["general"], 12345)
        report["code_canaries"] = asyncio.run(canaries())
        manifest, partitions, report["source_hashes"] = source_rows()
        report["dense"] = select(dense, manifest, partitions, {"math": 6, "ifeval": 6}, 4)
        report["moe"] = select(
            moe,
            manifest,
            partitions,
            {"math": 4, "ifeval": 4, "code": 2, "code_stdio": 2, "general-quality": 2, "general-quality_ref": 2},
            4,
        )
        report["passed"] = True
    finally:
        report["elapsed_seconds"] = time.time() - report["started"]
        workflow.write_json(Path("/output/colleague-preparation.json"), report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
