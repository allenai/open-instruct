"""Pin Think-SFT and validate the frozen Dolci prompts without resampling or rerendering."""

import argparse
import asyncio
import collections
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download
from scripts.miles import prepare_baseline_basket

from open_instruct.miles import run_data, workflow
from open_instruct.miles.run_spec import RunSpec

MODEL = "allenai/Olmo-3-7B-Think-SFT"
REVISION = "6ff857587e040d6d523a3d5f3a56e918f5401d66"
SOURCE = Path("/weka/oe-training-default/robertb/open-instruct/data/full-sft-basket-20260914")
# Pinned from docs/miles/measurements/full-sft-basket-20260914/preparation.json.
SOURCE_HASHES = {
    "eval-code.jsonl": "b4cd346c2dd853e68778225baa27dd56aff70a8dd6b061b379a0b4007f039e88",
    "eval-general.jsonl": "974fd34e043a37f482a9183473f035de5543741de8315b7833cb4fa0fbc1a0d0",
    "eval-ifeval.jsonl": "b3251e88df7701bf4daf93ebf44ad34407310a019e16a838765c87dbd4f5d626",
    "eval-math.jsonl": "a858ef1a095b8e9d0d41b684e65b40a3b052c28d351128f0b876e671d847403c",
    "train.jsonl": "fde6da774f735ea8d3720598f85ecbd613d5fcf0533f85dd5d8997fedbe93805",
    "verifiers.json": "89c377d9b6b201a37698e80396b8b557ccc4538b0e3c1f583e1db157ea9ddf8e",
}


def prepare_data(spec, tokenizer, source=SOURCE, expected_hashes=SOURCE_HASHES):
    output = Path(spec.data["prompt_data"]).parent
    if output.exists():
        raise ValueError(f"Immutable preparation output already exists: {output}; choose a new identity")
    inputs = {}
    for name, expected in expected_hashes.items():
        path = source / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen baseline artifact changed: {path}")
        inputs[str(path)] = actual
    registry = json.loads((source / "verifiers.json").read_text())
    limit = spec.compile().miles["rollout_max_prompt_len"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    counts, verifier_counts, held_out_ids, changed_tokens = {}, {}, {}, collections.Counter()
    identities, prompts = {}, {}
    try:
        for name in sorted(expected_hashes):
            if not name.endswith(".jsonl"):
                shutil.copyfile(source / name, temporary / name)
                continue
            split = Path(name).stem
            counts[split] = 0
            verifier_counts[split] = collections.Counter()
            identities[split], prompts[split] = set(), set()
            if split != "train":
                held_out_ids[split] = []
            with (source / name).open() as reader, (temporary / name).open("w") as writer:
                for line in reader:
                    row = json.loads(line)
                    metadata = row["metadata"]
                    old_hash = metadata.get("run_prompt_token_ids_sha256")
                    # Only tokenizer-dependent measurements change. Input text,
                    # labels, source IDs, verifier targets and row order stay intact.
                    run_data._verify_row(row, tokenizer, limit, registry)
                    changed_tokens[split] += old_hash != metadata["run_prompt_token_ids_sha256"]
                    identity = metadata["prepared_sample_id"]
                    identities[split].add(identity)
                    prompts[split].add(row["input"])
                    counts[split] += 1
                    verifier_counts[split][metadata["verifiers"][0]["name"]] += 1
                    if split != "train":
                        held_out_ids[split].append(identity)
                    writer.write(json.dumps(row) + "\n")
        for split in identities:
            if split != "train" and (identities["train"] & identities[split] or prompts["train"] & prompts[split]):
                raise ValueError(f"Training/held-out overlap: {split}")
        canaries = asyncio.run(prepare_baseline_basket.code_canaries())
        report = dict(
            passed=True,
            model=spec.model,
            hf_model=MODEL,
            hf_revision=REVISION,
            sources=inputs,
            counts=counts,
            verifier_counts={key: dict(value) for key, value in verifier_counts.items()},
            held_out_ids=held_out_ids,
            rendered_prompts_unchanged=True,
            changed_prompt_tokenizations=dict(changed_tokens),
            max_prompt_tokens=limit,
            code_canaries=canaries,
            outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in temporary.iterdir()},
        )
        workflow.write_json(temporary / "preparation.json", report)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    workflow.write_json("/output/preparation.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "held_out_ids"}), flush=True)


def prepare(spec):
    target = Path(spec.model["source"])
    if target.name != f"olmo3-think-sft-{REVISION}":
        raise ValueError("Think-SFT preparation requires the revision-specific checkpoint path")
    snapshot_download(
        MODEL,
        revision=REVISION,
        local_dir=target,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model", "LICENSE*", "README.md"],
    )
    index = json.loads((target / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if not shards or any(not (target / name).is_file() or not (target / name).stat().st_size for name in shards):
        raise ValueError("Pinned Think-SFT snapshot is missing indexed weight shards")
    checkpoint = dict(model=MODEL, revision=REVISION, shards=shards, identity=workflow.model_identity(target))
    workflow.write_json(Path(spec.output["root"]) / "checkpoint-preparation.json", checkpoint)
    workflow.write_json("/output/checkpoint-preparation.json", checkpoint)
    print(json.dumps(checkpoint), flush=True)
    prepare_data(spec, run_data._tokenizer(target))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    prepare(RunSpec.load(args.config))
