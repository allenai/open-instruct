"""Prepare one canonical GSM8K selection for both comparison arms, on CPU."""

import argparse
import copy
import hashlib
import importlib
import json
import struct
from pathlib import Path

DATASET = "ai2-adapt-dev/rlvr_gsm8k_zs"
REVISION = "93ffaae6cd2acb8f821f6d4712651320a889b1b9"
COUNTS = {"train": 400, "eval": 128}
CHECKPOINT_ROOT = Path("/weka/oe-training-default/robertb/olmo-miles/checkpoints")
CHECKPOINT_NAME = "olmoe3-kda-1.2b-dolci-think-sft-65536-router-bf16-autocast-v2"
HF_SOURCE = CHECKPOINT_ROOT / (CHECKPOINT_NAME + "-hf")
TEMPLATE_SOURCE = CHECKPOINT_ROOT / (CHECKPOINT_NAME + "-megatron/.olmo-miles/hf/chat_template.jinja")
TEMPLATE_SHA256 = "f5186d42d99c8a0445d37fd8a6c7ccf07fe3e24a29ce622d8bd245da9507b12b"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_immutable(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Refusing to replace different preparation artifact: {path}")
    else:
        path.write_bytes(raw)


def preparation_toml():
    text = 'schema_version = 1\nseed = 17\nshuffle = false\noutput_format = "jsonl"\n'
    for partition, count in COUNTS.items():
        split = "train" if partition == "train" else "test"
        text += (
            f'\n[[tasks]]\nname = "gsm8k-{partition}"\nprofile = "gsm8k"\n'
            f'partition = "{partition}"\nsplit = "{split}"\ncount = {count}\nrepeat = 1\n'
            f'source = {{kind = "huggingface", dataset = "{DATASET}", revision = "{REVISION}"}}\n'
        )
    return text


def derive_rows(native_rows, tokenizer, max_prompt_tokens=2048):
    """Render canonical messages once, proving native-chat and completion IDs agree."""
    rows, evidence = [], []
    for native in native_rows:
        messages = native["messages"]
        if not messages or messages[-1]["role"] != "user" or any(m["role"] == "assistant" for m in messages):
            raise ValueError("Canonical GSM8K prompts must end with a user and contain no reference assistant answer")
        label = native["ground_truth"]
        metadata = copy.deepcopy(native["metadata"])
        if not isinstance(label, str) or not label or not metadata.get("prepared_sample_id"):
            raise ValueError("Missing canonical string label or prepared_sample_id")
        if metadata.get("verifiers") != [{"name": "gsm8k", "target": label, "weight": 1.0}]:
            raise ValueError("Unexpected canonical GSM8K verifier schema")
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, tools=None, add_generation_prompt=True)
        native_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, tools=None, return_dict=False, add_generation_prompt=True
        )
        rendered_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if native_ids != rendered_ids:
            raise ValueError("Native MILES chat tokenization and Core rendered completion IDs differ")
        if len(rendered_ids) > max_prompt_tokens:
            raise ValueError("A selected prompt exceeds the shared prompt budget; no rows were silently filtered")
        metadata["query"] = messages[-1]["content"]
        metadata["source_id"] = native["id"]
        rows.append({"id": native["id"], "input": prompt, "label": label, "metadata": metadata})
        evidence.append(
            {
                "id": native["id"],
                "prepared_sample_id": metadata["prepared_sample_id"],
                "source": metadata.get("source"),
                "prompt_sha256": digest(prompt.encode()),
                "token_ids_sha256": digest(json_bytes(rendered_ids)),
                "prompt_tokens": len(rendered_ids),
            }
        )
    if len({item["prepared_sample_id"] for item in evidence}) != len(rows):
        raise ValueError("Duplicate canonical sample IDs")
    return rows, evidence


def prepare_descriptor(root):
    template = TEMPLATE_SOURCE.read_bytes().removesuffix(b"\n")
    if digest(template) != TEMPLATE_SHA256:
        raise ValueError("Checkpoint-native template hash differs from the shared comparison contract")
    shards = sorted(HF_SOURCE.glob("*.safetensors"))
    if not shards or not (HF_SOURCE / "config.json").is_file():
        raise ValueError("SFT checkpoint shards/config missing")
    hf = root / "hf"
    hf.mkdir(parents=True, exist_ok=True)
    for source in HF_SOURCE.iterdir():
        if source.is_file() and source.name != "chat_template.jinja":
            destination = hf / source.name
            if destination.is_symlink():
                if destination.resolve() != source.resolve():
                    raise ValueError(f"Descriptor points to another checkpoint: {destination}")
            elif destination.exists():
                raise ValueError(f"Descriptor file is not a source symlink: {destination}")
            else:
                destination.symlink_to(source)
    write_immutable(hf / "chat_template.jinja", template)
    headers = {}
    for shard in shards:
        with shard.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < size < shard.stat().st_size - 8:
                raise ValueError(f"Invalid safetensors header: {shard}")
            headers[shard.name] = digest(stream.read(size))
    return {
        "source": str(HF_SOURCE),
        "config_sha256": digest((hf / "config.json").read_bytes()),
        "source_header_sha256": headers,
        "template_sha256": TEMPLATE_SHA256,
    }


def prepare(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "preparation.json").exists():
        return verify_preparation(root)
    descriptor = prepare_descriptor(root)
    tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(
        root / "hf", trust_remote_code=True
    )
    if digest(tokenizer.chat_template.encode()) != TEMPLATE_SHA256:
        raise ValueError("Tokenizer did not select the pinned checkpoint-native template")
    config = root / "tasks.toml"
    write_immutable(config, preparation_toml().encode())
    native_preparer = importlib.import_module("olmo_miles.rl.rl_prepare")
    manifest = native_preparer.prepare_rl_datasets(config, output_dir=root / "baseline")
    native_preparer.load_rl_manifest(manifest)
    partitions = {}
    for partition, count in COUNTS.items():
        native = [json.loads(line) for line in (root / "baseline" / f"{partition}.jsonl").read_text().splitlines()]
        if len(native) != count:
            raise ValueError(f"Canonical {partition} count differs: {len(native)} != {count}")
        rows, evidence = derive_rows(native, tokenizer)
        write_immutable(root / f"{partition}.jsonl", b"".join(json.dumps(row).encode() + b"\n" for row in rows))
        partitions[partition] = {"records": count, "rows": evidence}
    train_prompts = {row["prompt_sha256"] for row in partitions["train"]["rows"]}
    if train_prompts.intersection(row["prompt_sha256"] for row in partitions["eval"]["rows"]):
        raise ValueError("Selected official train/test prompts overlap")
    write_immutable(
        root / "verifiers.json", json_bytes({"gsm8k": {"factory": "open_instruct.ground_truth_utils.GSM8KVerifier"}})
    )
    paths = [
        "tasks.toml",
        "train.jsonl",
        "eval.jsonl",
        "verifiers.json",
        "baseline/rl-manifest.json",
        "baseline/train.jsonl",
        "baseline/eval.jsonl",
        "hf/config.json",
        "hf/chat_template.jinja",
        "hf/tokenizer.json",
        "hf/tokenizer_config.json",
    ]
    report = {
        "schema_version": 1,
        "dataset": DATASET,
        "revision": REVISION,
        "seed": 17,
        "shuffle": False,
        "selection": "One canonical olmo_miles.rl.rl_prepare selection; Core rows derived in identical order",
        "tokenization": "native chat tokenize=True equals rendered encode(add_special_tokens=False) for every row",
        "eval_split": "test",
        "baseline_manifest": "baseline/rl-manifest.json",
        "descriptor": descriptor,
        "partitions": partitions,
        "files": {name: digest((root / name).read_bytes()) for name in paths},
    }
    write_immutable(root / "preparation.json", json_bytes(report))
    return verify_preparation(root)


def verify_preparation(root):
    """Verify the frozen shared files and counts without loading weights or CUDA."""
    root = Path(root)
    report = json.loads((root / "preparation.json").read_text())
    if report.get("schema_version") != 1 or report.get("revision") != REVISION:
        raise ValueError("Unsupported shared preparation schema or dataset revision")
    required = {
        "tasks.toml",
        "train.jsonl",
        "eval.jsonl",
        "verifiers.json",
        "baseline/rl-manifest.json",
        "baseline/train.jsonl",
        "baseline/eval.jsonl",
        "hf/config.json",
        "hf/chat_template.jinja",
        "hf/tokenizer.json",
        "hf/tokenizer_config.json",
    }
    if set(report["files"]) != required:
        raise ValueError("Missing or unexpected shared artifact hashes")
    for name, expected in report["files"].items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError(f"Shared artifact SHA256 differs: {name}")
    if report["files"]["hf/chat_template.jinja"] != TEMPLATE_SHA256:
        raise ValueError("Shared template differs from the pinned template")
    for partition, count in COUNTS.items():
        rows = [json.loads(line) for line in (root / f"{partition}.jsonl").read_text().splitlines()]
        native = [json.loads(line) for line in (root / "baseline" / f"{partition}.jsonl").read_text().splitlines()]
        evidence = report["partitions"][partition]
        if (
            len(rows) != count
            or len(native) != count
            or evidence["records"] != count
            or len(evidence["rows"]) != count
        ):
            raise ValueError(f"Shared {partition} count differs")
        for core, baseline, proof in zip(rows, native, evidence["rows"], strict=True):
            if core["id"] != baseline["id"] or core["id"] != proof["id"] or core["label"] != baseline["ground_truth"]:
                raise ValueError("Shared canonical row order/identity/labels differ")
            if core["metadata"]["prepared_sample_id"] != baseline["metadata"]["prepared_sample_id"]:
                raise ValueError("Shared prepared sample IDs differ")
            if digest(core["input"].encode()) != proof["prompt_sha256"]:
                raise ValueError("Shared rendered prompt differs")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    report = verify_preparation(args.root) if args.verify_only else prepare(args.root)
    print(json.dumps({"root": str(args.root), "records": COUNTS, "files": report["files"]}), flush=True)


if __name__ == "__main__":
    main()
