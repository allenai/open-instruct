"""Export audited warmup parameters, without its optimizer history, for native SFT."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import torch as safe_torch
from tokenizers import Tokenizer

MATRICES = ("embeddings.weight", "lm_head.w_out.weight")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()


def export(root, report_path, pins_path, tokenizer_path, arm, source_beaker, output):
    report = json.loads(report_path.read_text())
    receipt = report["arms"][arm]
    checkpoint = root / arm / "step100.pt"
    if digest(checkpoint) != receipt["checkpoint_sha256"] or digest(pins_path) != report["base_pins_sha256"]:
        raise ValueError("Warmup checkpoint or base pins changed")
    if not receipt["no_update_logits_exact"] or not receipt["frozen_original_parameters_unchanged"]:
        raise ValueError("Warmup did not pass its integrity checks")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if state["next_step"] != 100 or state["base"] != report["base"] or state["data_sha256"] != report["data_sha256"]:
        raise ValueError("Warmup cursor/base/data mismatch")
    if set(state["adapters"]) != set(MATRICES) or receipt["scope"] not in ("rows", "full"):
        raise ValueError("Unexpected warmup parameters")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    atomic = receipt["encoding"] == "atomic"
    if receipt["encoding"] not in ("atomic", "original"):
        raise ValueError("Unknown encoding")
    if atomic != (tokenizer.token_to_id("<think>") == 100256 and tokenizer.token_to_id("</think>") == 100270):
        raise ValueError("Tokenizer disagrees with warmup encoding")
    pins = json.loads(pins_path.read_text())
    entries, values = {}, {}
    for name, adapter in state["adapters"].items():
        rows = receipt["scope"] == "rows"
        if rows and (not atomic or adapter["indices"].tolist() != [100256, 100270]):
            raise ValueError("Wrong warmed row IDs")
        value = adapter["rows" if rows else "master"]
        if value.dtype != torch.float32 or tuple(value.shape) != ((2, 1024) if rows else (100352, 1024)):
            raise ValueError("Wrong warmup shape or precision")
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite warmup parameters")
        values[name] = value.contiguous()
        entries[name] = {
            "scope": receipt["scope"],
            "tensor_sha256": tensor_digest(value),
            "base_master_sha256": pins["matrices"][name]["master"],
        }
        if rows:
            entries[name]["rows"] = [100256, 100270]
    output.mkdir(parents=True, exist_ok=False)
    payload = output / "matrices.safetensors"
    safe_torch.save_file(values, str(payload))
    loaded = safe_torch.load_file(str(payload))
    if any(not torch.equal(loaded[name], values[name]) for name in MATRICES):
        raise ValueError("Export changed FP32 values")
    manifest = {
        "version": 1,
        "optimizer_policy": "fresh-sft",
        "base": str(Path(state["base"]).parent),
        "encoding": receipt["encoding"],
        "tokenizer_vocab_sha256": hashlib.sha256(
            json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "file": payload.name,
        "file_sha256": digest(payload),
        "matrices": entries,
        "source": {
            "beaker": source_beaker,
            "arm": arm,
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "report_sha256": digest(report_path),
            "base_pins_sha256": digest(pins_path),
            "data_sha256": state["data_sha256"],
            "updates": 100,
            "input_tokens": report["training_tokens"][receipt["encoding"]],
            "supervised_targets": report["supervised_targets"][receipt["encoding"]],
        },
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return digest(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "report", "pins", "tokenizer", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--source-beaker", required=True)
    args = parser.parse_args()
    print(export(args.root, args.report, args.pins, args.tokenizer, args.arm, args.source_beaker, args.output))


if __name__ == "__main__":
    main()
