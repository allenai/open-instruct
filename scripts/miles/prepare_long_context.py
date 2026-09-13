"""Select natural long prompts from a pinned unfiltered Dolci Think source on Saturn."""

import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from scripts.miles.prepare_judge_exercise import CODE_URL

from open_instruct.miles import run_data, workflow

DATASET = "allenai/Dolci-Think-RL-7B"
REVISION = "0fb6466d31ef3a9dd16985ef635e6429e05a6491"


def aligned(value):
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            value = json.loads(value)
        except ValueError:
            value = ast.literal_eval(value)
    return value if isinstance(value, list) else [value]


def canonical(source, index, tokenizer, dataset=DATASET, revision=REVISION):
    names = aligned(source.get("verifier_source") or source.get("dataset"))
    targets = aligned(source["ground_truth"])
    if len(names) != 1 or len(targets) != 1 or names[0] not in ("math", "ifeval", "code", "code_stdio"):
        return None
    if source.get("messages"):
        messages = run_data._messages(source, strip_answer=True)
        content = "\n".join(m["content"] for m in messages)
    else:
        prompt = source["prompt"].strip()
        if not prompt.lower().startswith("user:"):
            raise ValueError("Expected Dolci user:-prefixed prompt")
        content = prompt.split(":", 1)[1].strip()
        messages = [{"role": "user", "content": content}]
    rendered = run_data._render(messages, tokenizer, tokenizer.chat_template)
    identity = hashlib.sha256(" ".join(content.split()).encode()).hexdigest()
    row = {
        "input": rendered,
        "label": source["ground_truth"],
        "metadata": {
            "prepared_sample_id": identity,
            "source": {"dataset": dataset, "revision": revision, "index": index, "custom_id": source.get("custom_id")},
            "query": content,
            "verifiers": [{"name": names[0], "target": targets[0], "weight": 1.0}],
        },
    }
    return row, len(tokenizer.encode(rendered, add_special_tokens=False))


def prepare(model, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new immutable long-context fixture directory")
    config = json.loads((Path(model) / "config.json").read_text())
    if config.get("max_position_embeddings", 0) < 16384:
        raise ValueError("Checkpoint does not advertise a 16K context")
    tokenizer = run_data._tokenizer(Path(model))
    rows = load_dataset(
        DATASET,
        revision=REVISION,
        split="train",
        cache_dir="/weka/oe-training-default/robertb/olmo-miles/dataset-cache/huggingface/datasets",
    )
    selected = {"long": [], "short": []}
    counts, seen = collections.Counter(), set()
    for index, source in enumerate(rows):
        result = canonical(source, index, tokenizer)
        if result is None:
            continue
        row, length = result
        key = row["metadata"]["prepared_sample_id"]
        if key in seen:
            continue
        seen.add(key)
        domain = row["metadata"]["verifiers"][0]["name"]
        counts[f"{domain}:total"] += 1
        kind = "long" if 4096 < length <= 8192 else "short" if length <= 2048 and domain == "math" else None
        if kind:
            counts[kind] += 1
            if len(selected[kind]) < 10:
                row["metadata"]["readiness_length_class"] = kind
                selected[kind].append(row)
        if all(len(v) == 10 for v in selected.values()):
            break
    extra_source = {
        "dataset": "hamishivi/code_rlvr_mixture_dpo",
        "revision": "0c37776831a2e956935bcabafa3915bcdf353a30",
    }
    if len(selected["long"]) < 10:
        print("Dolci has no sufficient long-input coverage; scanning pinned Olmo 3 code mixture", flush=True)
        extra = load_dataset(
            extra_source["dataset"],
            revision=extra_source["revision"],
            split="train",
            cache_dir="/weka/oe-training-default/robertb/olmo-miles/dataset-cache/huggingface/datasets",
        )
        for index, source in enumerate(extra):
            result = canonical(source, index, tokenizer, **extra_source)
            if result is None:
                continue
            row, length = result
            key = row["metadata"]["prepared_sample_id"]
            if key in seen:
                continue
            seen.add(key)
            counts["code_mixture_scanned"] += 1
            if 4096 < length <= 8192:
                row["metadata"]["readiness_length_class"] = "long"
                selected["long"].append(row)
                counts["code_mixture_long"] += 1
            if len(selected["long"]) == 10:
                break
    if any(len(v) != 10 for v in selected.values()):
        raise ValueError(f"Insufficient natural long/short fixtures: {dict(counts)}")
    registry = {name: {"factory": factory} for name, factory in run_data.FACTORIES.items()}
    for name in ("code", "code_stdio"):
        registry[name] = {
            "factory": "open_instruct.miles.code_rewards.CodeVerifier",
            "config": {"api_url": CODE_URL, "stdio": name == "code_stdio"},
        }
    partitions = {"eval": [], "train": []}
    for index in range(10):
        split = "eval" if index < 2 else "train"
        for kind in ("long", "short"):
            row = selected[kind][index]
            run_data._verify_row(row, tokenizer, 8192, registry)
            partitions[split].append(row)
    output.mkdir(parents=True)
    for split, partition in partitions.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in partition))
    workflow.write_json(output / "verifiers.json", registry)
    report = {
        "passed": True,
        "dataset": DATASET,
        "revision": REVISION,
        "model": str(model),
        "source_rows": len(rows),
        "additional_source": extra_source,
        "scanned_counts": dict(counts),
        "lengths": {
            split: [
                {
                    "id": r["metadata"]["prepared_sample_id"],
                    "tokens": r["metadata"]["run_prompt_tokens"],
                    "domain": r["metadata"]["verifiers"][0]["name"],
                }
                for r in partition
            ]
            for split, partition in partitions.items()
        },
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()},
        "limits": {"prompt": 8192, "response": 8192, "context": 16384},
        "synthetic_padding": False,
    }
    workflow.write_json(output / "preparation.json", report)
    workflow.write_json(Path("/output/long-context-preparation.json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        result = prepare(args.model, args.output)
    except Exception as error:
        workflow.write_json(
            Path("/output/long-context-preparation.json"),
            {"passed": False, "error": f"{type(error).__name__}: {error}"},
        )
        raise
    print(json.dumps(result, indent=2))
