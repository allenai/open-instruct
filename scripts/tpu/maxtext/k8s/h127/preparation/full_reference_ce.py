"""Source HF model reference CE on the exact packed TPU first batch, split by document."""

import argparse
import gzip
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributed import checkpoint as dcp
from transformers import AutoModelForCausalLM

SOURCE = Path("/weka/olmo-3p5-checkpoints/protected/olmo35-14t600b-mt100b-lc-sft-20261014/lc/step5961/olmo-core")
HF = "/weka/olmo-3p5-checkpoints/scratch/olmo35-14t600b-mt100b-lc-sft-20261014/olmo35-14t600b-mt100b-lc-sft-20261014-lc/emo/step5961/hf"
OUT = Path("/output")
ROWS = [(100256, [14023, 771, 29]), (100270, [524, 27963, 29])]


def capture_base_metadata(model):
    destination = OUT / "hf-reference"
    destination.mkdir(exist_ok=False)
    for path in Path(HF).iterdir():
        if path.is_file() and path.suffix in (".json", ".py", ".jinja") and not path.name.endswith(".index.json"):
            shutil.copy2(path, destination / path.name)
        elif path.is_dir() and path.name == "chat_templates":
            shutil.copytree(path, destination / path.name)
    config = json.loads((destination / "config.json").read_text())
    source_vocab = config["vocab_size"]
    config["vocab_size"] = 100352
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    shapes, hashes = {}, {}
    for name, value in model.state_dict().items():
        assert value.device.type == "cpu" and value.is_floating_point(), (name, value.dtype)
        array = value.detach().to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
        shapes[name] = list(array.shape)
        hashes[name] = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
    (destination / "reference-shapes.json").write_text(json.dumps(shapes, sort_keys=True))
    (destination / "expected-seeded-bf16-sha256.json").write_text(json.dumps(hashes, sort_keys=True))
    (destination / "metadata-receipt.json").write_text(
        json.dumps(
            {
                "source": HF,
                "source_vocab": source_vocab,
                "export_vocab": 100352,
                "tensors": len(shapes),
                "values": sum(value.numel() for value in model.state_dict().values()),
                "scope": "Source HF model after the registered two-row atomic initialization; all other HF weights unchanged",
            },
            indent=2,
        )
    )
    print("HF-METADATA-AND-HASHES", len(shapes), flush=True)


def load_model(capture_metadata=False):
    reader = dcp.FileSystemReader(str(SOURCE / "model_and_optim"))
    original_read = reader.read_metadata

    def read_metadata(*args, **kwargs):
        metadata = original_read(*args, **kwargs)
        for value in metadata.storage_data.values():
            if not hasattr(value, "transform_descriptors"):
                object.__setattr__(value, "transform_descriptors", None)
        return metadata

    reader.read_metadata = read_metadata
    metadata = reader.read_metadata()
    keys = ["module.embeddings.weight.main", "module.lm_head.w_out.weight.main"]
    state = {key: torch.empty(tuple(metadata.state_dict_metadata[key].size), dtype=torch.float32) for key in keys}
    dcp.load(state, storage_reader=reader)
    matrices = [state[key].reshape(100352, 1024) for key in keys]
    model = AutoModelForCausalLM.from_pretrained(
        HF, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    matrix_refs = [model.get_input_embeddings().weight, model.get_output_embeddings().weight]
    for expected, actual in zip(matrices, matrix_refs, strict=True):
        assert torch.equal(expected[: actual.shape[0]].to(torch.bfloat16), actual.detach()), (
            "HF embedding/head differs from native checkpoint"
        )
    model.resize_token_embeddings(100352, mean_resizing=False)
    matrix_refs = [model.get_input_embeddings().weight, model.get_output_embeddings().weight]
    initial_rows = {}
    with torch.no_grad():
        for key, master, target in zip(keys, matrices, matrix_refs, strict=True):
            bf16 = master.to(torch.bfloat16)
            for row, pieces in ROWS:
                bf16[row] = bf16[pieces].float().mean(0).to(torch.bfloat16)
            target.copy_(bf16)
            initial_rows[key] = hashlib.sha256(bf16[[row for row, _ in ROWS]].float().numpy().tobytes()).hexdigest()
    del state, matrices, matrix_refs
    if capture_metadata:
        capture_base_metadata(model)
    model = model.cuda().eval()
    return model, initial_rows


def score_batch(model, arm):
    rows = []
    total, total_tokens = 0.0, 0
    started = time.monotonic()
    with torch.inference_mode():
        for host in range(8):
            batch = np.load(f"/probe/full-replay/{arm}/h{host}.step1.npz")
            for row in range(batch["inputs"].shape[0]):
                segments = batch["inputs_segmentation"][row]
                loss_sum, count = 0.0, 0
                for segment in sorted(set(segments[segments > 0].tolist())):
                    idx = np.flatnonzero(segments == segment)
                    positions = batch["inputs_position"][row][idx]
                    assert positions[0] == 0 and (np.diff(positions) == 1).all()
                    ids = torch.tensor(batch["inputs"][row][idx].astype(np.int64), device="cuda")[None]
                    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
                    targets = torch.tensor(batch["targets"][row][idx].astype(np.int64), device="cuda")
                    mask = torch.tensor(batch["targets_segmentation"][row][idx] > 0, device="cuda")
                    for offset in range(0, len(idx), 1024):
                        logits = model.lm_head(hidden[offset : offset + 1024]).float()
                        nll = torch.nn.functional.cross_entropy(
                            logits, targets[offset : offset + 1024], reduction="none"
                        )
                        loss_sum += float((nll * mask[offset : offset + 1024]).sum())
                        count += int(mask[offset : offset + 1024].sum())
                assert count > 0
                total += loss_sum
                total_tokens += count
                rows.append({"host": host, "row": row, "loss_tokens": count, "ce": loss_sum / count})
                print("REF-ROW", arm, host, row, count, loss_sum / count, flush=True)
    expected_tokens = sum(
        json.loads(Path(f"/probe/full-replay/{arm}/h{host}.json").read_text())["batches"][0]["loss_tokens"]
        for host in range(8)
    )
    assert total_tokens == expected_tokens
    return {
        "ce": total / total_tokens,
        "loss_tokens": total_tokens,
        "rows": rows,
        "seconds": time.monotonic() - started,
    }


def score_openers(model):
    with gzip.open("/probe/math-opener-panel.jsonl.gz", "rt") as stream:
        panel = [json.loads(line) for line in stream]
    family = [r["id"] for r in json.loads(Path("/probe/newline-token-family.json").read_text())]
    assert len(panel) == len({r["qualified_id"] for r in panel}) == 1000
    assert len(set(family)) == 48 and not set(family) & {33413, 36539}
    repeats = []

    def forward(ids):
        hidden = model.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).last_hidden_state[:, -1]
        return model.lm_head(hidden).float().softmax(-1)[0]

    with (OUT / "step0-opener-probabilities.jsonl").open("w") as stream, torch.inference_mode():
        for index, record in enumerate(panel):
            ids = record["input_ids"]
            assert ids[-1] == 100256
            probabilities = forward(ids)
            newline = float(probabilities[family].sum())
            okay, space_okay = (float(probabilities[t]) for t in (33413, 36539))
            result = {
                "qualified_id": record["qualified_id"],
                "split": record["split"],
                "input_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                "newline_family": newline,
                "Okay": okay,
                "space_Okay": space_okay,
                "remainder": float(probabilities.sum()) - newline - okay - space_okay,
                "token_probabilities": {str(token): float(probabilities[token]) for token in family + [33413, 36539]},
                "step": 0,
            }
            assert (
                result["remainder"] >= -1e-6
                and abs(sum(result[key] for key in ("newline_family", "Okay", "space_Okay", "remainder")) - 1) < 1e-5
            )
            stream.write(json.dumps(result) + "\n")
            if index < 16:
                repeat = forward(ids)
                repeats.append(
                    {
                        "qualified_id": record["qualified_id"],
                        "max_absolute_probability_difference": float((repeat - probabilities).abs().max()),
                        "total_variation": float((repeat - probabilities).abs().sum() / 2),
                    }
                )
            if (index + 1) % 100 == 0:
                stream.flush()
                print("OPENER-PANEL", index + 1, "of", len(panel), flush=True)
    return {"n": len(panel), "repeated_identical_forwards": repeats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("all", "reference", "step0"), default="all")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    model, initial_rows = load_model(capture_metadata=args.mode in ("all", "step0"))
    receipt = {"source": HF, "initial_rows": initial_rows, "arms": {}}
    if args.mode in ("all", "reference"):
        for arm in ("control", "uniform"):
            receipt["arms"][arm] = score_batch(model, arm)
            (OUT / "full-reference-ce.json").write_text(json.dumps(receipt, indent=2))
    if args.mode in ("all", "step0"):
        receipt["step0_panel"] = score_openers(model)
    receipt["provenance_sha256"] = {
        name: hashlib.sha256((Path("/probe") / name).read_bytes()).hexdigest()
        for name in ("math-opener-panel.jsonl.gz", "newline-token-family.json", "full-replay/qualification.json")
        if (Path("/probe") / name).exists()
    }
    (OUT / "full-reference-ce.json").write_text(json.dumps(receipt, indent=2))
    print("FULL-REFERENCE-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
