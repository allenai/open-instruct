"""Diagnose historical proof drift with the actual MILES dataset and serving tokenizer."""

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from miles.rollout.generate_utils import generate_endpoint_utils
from miles.utils import data, processing_utils
from scripts.miles.prepare_gsm8k_parity import digest, json_bytes
from sglang.srt.utils.hf_transformers import tokenizer as serving_tokenizer
from tokenizers import Tokenizer
from transformers import AutoTokenizer


def compare_partition(path, hf, proofs):
    native = processing_utils.load_tokenizer(str(hf), trust_remote_code=True)
    bare = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    raw = Tokenizer.from_file(str(hf / "tokenizer.json"))
    serving = serving_tokenizer.get_tokenizer(str(hf), trust_remote_code=True)
    dataset = data.Dataset(
        path=str(path),
        tokenizer=native,
        processor=None,
        max_length=None,
        prompt_key="input",
        label_key="label",
        metadata_key="metadata",
        apply_chat_template=False,
    )
    if len(dataset.samples) != len(proofs):
        raise ValueError("Actual dataset reader changed the frozen question count")
    state = SimpleNamespace(tokenizer=native, processor=None)
    rows, differences = [], []
    counts = {"bare_transformers": 0, "checkpoint_json": 0, "sglang": 0, "old_proof": 0}
    for sample, proof in zip(dataset.samples, proofs, strict=True):
        identity = sample.metadata["prepared_sample_id"]
        if identity != proof["prepared_sample_id"]:
            raise ValueError("Actual dataset reader changed question identity/order")
        ids = generate_endpoint_utils.compute_prompt_ids_from_sample(state, sample)
        alternatives = {
            "bare_transformers": bare.encode(sample.prompt, add_special_tokens=False),
            "checkpoint_json": raw.encode(sample.prompt, add_special_tokens=False).ids,
            "sglang": serving.encode(sample.prompt, add_special_tokens=False),
        }
        row = dict(proof)
        row.update(token_ids_sha256=digest(json_bytes(ids)), prompt_tokens=len(ids))
        rows.append(row)
        mismatch = {name: other != ids for name, other in alternatives.items()}
        mismatch["old_proof"] = row["token_ids_sha256"] != proof["token_ids_sha256"]
        for name, differs in mismatch.items():
            counts[name] += differs
        if any(mismatch.values()) and len(differences) < 3:
            differences.append(
                {
                    "id": identity,
                    "mismatch": mismatch,
                    "runtime_token_ids": ids,
                    "bare_token_ids": alternatives["bare_transformers"],
                }
            )
    return {
        "count": len(rows),
        "mismatch_counts": counts,
        "examples": differences,
        "max_prompt_tokens": max(row["prompt_tokens"] for row in rows),
        "runtime_proofs": rows,
        "valid": counts["checkpoint_json"] == counts["sglang"] == 0,
        "pre_tokenizer_types": {
            "bare": type(bare.backend_tokenizer.pre_tokenizer).__name__,
            "runtime": type(native.backend_tokenizer.pre_tokenizer).__name__,
            "checkpoint": type(raw.pre_tokenizer).__name__,
        },
    }


def diagnose(root, output):
    output.mkdir(parents=True, exist_ok=True)
    fixture = output / "fixture"
    (fixture / "hf").mkdir(parents=True)
    for path in (root / "hf").iterdir():
        if path.is_file() and path.suffix != ".safetensors":
            shutil.copyfile(path, fixture / "hf" / path.name)
    for name in (
        "train.jsonl",
        "eval.jsonl",
        "preparation.json",
        "offline/prompts.jsonl",
        "offline/token-proofs.json",
        "offline/requests.jsonl",
    ):
        target = fixture / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    preparation = json.loads((fixture / "preparation.json").read_text())
    sources = {
        "train": ("train.jsonl", preparation["partitions"]["train"]["rows"]),
        "eval": ("eval.jsonl", preparation["partitions"]["eval"]["rows"]),
        "offline": ("offline/prompts.jsonl", json.loads((fixture / "offline/token-proofs.json").read_text())),
    }
    report = {"root": str(root), "partitions": {}}
    for name, (path, proofs) in sources.items():
        report["partitions"][name] = compare_partition(fixture / path, fixture / "hf", proofs)
        print("TOKENIZATION_PARTITION", name, report["partitions"][name]["mismatch_counts"], flush=True)
    report["valid"] = all(partition["valid"] for partition in report["partitions"].values())
    (output / "tokenization.json").write_bytes(json_bytes(report))
    if not report["valid"]:
        raise ValueError("MILES tokenizer differs from checkpoint JSON or SGLang")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    diagnose(args.root, args.output)


if __name__ == "__main__":
    main()
