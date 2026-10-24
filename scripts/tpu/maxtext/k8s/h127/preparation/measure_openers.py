"""Measure raw post-think probabilities on a frozen panel from a trained HF export.

The forward and category calculation match H127's completed step-zero probe.
No generation, logit filtering, token initialization or optimizer updates occur.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

PANELS = {
    "96d18a968061fa88aae38ad1dd69ac297345db5ada3994522b6ca5de034dfecc": 1000,
    "99eceb44d123b0f653b80b9989c0ee4ec6de11164aec23c763f8155302d2653f": 5000,
}
FAMILY_SHA = "f3b6b0504d6d77eb6bb42505c4a49e8cad2ca5059ba9a02e14fefdc00dd0ccf0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=(0, 50, 512, 2048), required=True)
    parser.add_argument(
        "--arm",
        choices=("atomic-original", "atomic-presentation-balanced", "shared-base", "infra-smoke"),
        required=True,
    )
    parser.add_argument("--pilot-items", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel_sha = hashlib.sha256(args.panel.read_bytes()).hexdigest()
    assert panel_sha in PANELS
    assert hashlib.sha256(args.family.read_bytes()).hexdigest() == FAMILY_SHA
    assert (args.step == 0) == (args.arm == "shared-base")
    assert (args.step == 50) == (args.arm == "infra-smoke")
    assert (args.pilot_items is not None) == (args.step == 50)
    with gzip.open(args.panel, "rt") as stream:
        panel = [json.loads(line) for line in stream]
    assert len(panel) == PANELS[panel_sha]
    assert len({(r.get("variant", "original"), r["qualified_id"]) for r in panel}) == len(panel)
    assert len({r["qualified_id"] for r in panel}) == 1000
    if args.pilot_items is not None:
        assert 16 <= args.pilot_items <= 32
        panel = panel[: args.pilot_items]
    family = [r["id"] for r in json.loads(args.family.read_text())]
    assert len(set(family)) == 48 and not set(family) & {33413, 36539}
    export = json.loads((args.model / "export-receipt.json").read_text())
    with (args.model / "model.safetensors").open("rb") as stream:
        weight_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    assert weight_sha == export["model_safetensors_sha256"]
    assert json.loads((args.model / "config.json").read_text())["vocab_size"] == 100352
    args.output.mkdir(parents=True, exist_ok=False)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        .cuda()
        .eval()
    )

    def forward(ids):
        hidden = model.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).last_hidden_state[:, -1]
        return model.lm_head(hidden).float().softmax(-1)[0]

    repeats = []
    with (args.output / "opener-probabilities.jsonl").open("w") as stream, torch.inference_mode():
        for index, record in enumerate(panel):
            ids = record["input_ids"]
            assert ids[-1] == 100256
            probabilities = forward(ids)
            newline = float(probabilities[family].sum())
            okay, space_okay = (float(probabilities[t]) for t in (33413, 36539))
            result = {
                "qualified_id": record["qualified_id"],
                "variant": record.get("variant", "original"),
                "split": record["split"],
                "input_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                "newline_family": newline,
                "Okay": okay,
                "space_Okay": space_okay,
                "remainder": float(probabilities.sum()) - newline - okay - space_okay,
                "token_probabilities": {str(token): float(probabilities[token]) for token in family + [33413, 36539]},
                "step": args.step,
                "arm": args.arm,
            }
            assert result["remainder"] >= -1e-6
            assert abs(sum(result[k] for k in ("newline_family", "Okay", "space_Okay", "remainder")) - 1) < 1e-5
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            if len(repeats) < 16 and result["variant"] == "original":
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
    metadata_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.model.iterdir()
        if p.is_file() and p.suffix in (".json", ".py", ".jinja")
    }
    receipt = {
        "arm": args.arm,
        "step": args.step,
        "model": str(args.model),
        "export_receipt": export,
        "model_safetensors_sha256": weight_sha,
        "model_metadata_sha256": metadata_hashes,
        "panel_sha256": panel_sha,
        "family_sha256": FAMILY_SHA,
        "rows": len(panel),
        "qualification_only": args.step == 50,
        "repeated_identical_forwards": repeats,
        "scope": "Raw FP32 softmax on BF16 HF forward; before temperature, top-p or sampling",
    }
    (args.output / "measurement-receipt.json").write_text(json.dumps(receipt, indent=2))
    print("OPENER-MEASUREMENT-COMPLETE", args.arm, args.step, len(panel), flush=True)


if __name__ == "__main__":
    main()
