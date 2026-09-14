"""Audit the tiny sampled-token update and complete its HF model export."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors import torch as safetensors_torch

from open_instruct.miles import workflow


def audit_training(root, num_rollouts, coefficient):
    records = []
    for rollout_id in range(num_rollouts):
        path = root / f"debug/train_data/{rollout_id}_0.pt"
        data = torch.load(path, map_location="cpu", weights_only=False)["rollout_data"]
        student, teacher, advantages = (data[key] for key in ("log_probs", "teacher_log_probs", "advantages"))
        if not student or not len(student) == len(teacher) == len(advantages):
            raise ValueError("Missing or misaligned OPD training data")
        errors, signals = [], []
        for learner, target, advantage in zip(student, teacher, advantages):
            if not learner.shape == target.shape == advantage.shape:
                raise ValueError("OPD tensors have different shapes")
            for value in (learner, target, advantage):
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite OPD tensor")
            expected = coefficient * (target - learner)
            errors.append(float((advantage - expected).abs().max()))
            signals.append(float(expected.abs().max()))
        if max(errors) > 1e-5 or max(signals) == 0:
            raise ValueError("Training advantages do not contain the expected nonzero teacher signal")
        records.append(
            {
                "rollout_id": rollout_id,
                "samples": len(student),
                "max_advantage_error": max(errors),
                "max_abs_opd_signal": max(signals),
            }
        )
    return records


def complete_export(root, base, rollout_id):
    """Preserve frozen vision/MTP weights omitted by the native language-only exporter."""
    export = root / f"hf-{rollout_id}"
    if not (export / ".complete").exists():
        raise ValueError(f"Native HF export did not complete: {export}")
    index_path = export / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    original = json.loads((base / "model.safetensors.index.json").read_text())
    original_map, weight_map = original["weight_map"], index["weight_map"]
    changed, fp32_a_logs = 0, 0
    for shard in sorted(set(weight_map.values())):
        tensors = safetensors_torch.load_file(export / shard)
        for name, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Export contains nonfinite weights: {name}")
            if name.endswith("A_log"):
                if tensor.dtype != torch.float32:
                    raise ValueError(f"Export lost A_log precision: {name}")
                fp32_a_logs += 1
            if name in original_map:
                with safe_open(base / original_map[name], framework="pt", device="cpu") as source:
                    old = source.get_tensor(name)
                changed += int(not torch.equal(old, tensor))
    if changed == 0 or fp32_a_logs == 0:
        raise ValueError("Export lacks changed weights or FP32 A_log tensors")
    missing = sorted(set(original_map) - set(weight_map))
    # Missing language weights would be an export failure, not frozen extras.
    if any(name.startswith("model.language_model.") for name in missing):
        raise ValueError("Native export omitted language model weights")
    frozen = {}
    for name in missing:
        with safe_open(base / original_map[name], framework="pt", device="cpu") as source:
            frozen[name] = source.get_tensor(name)
    (export / ".complete").unlink()
    if frozen:
        shard = "frozen-base.safetensors"
        safetensors_torch.save_file(frozen, export / shard)
        weight_map.update({name: shard for name in frozen})
        index["metadata"]["total_size"] += sum(t.numel() * t.element_size() for t in frozen.values())
        workflow.write_json(index_path, index)
    result = {
        "path": str(export),
        "changed_tensors": changed,
        "fp32_a_log_tensors": fp32_a_logs,
        "frozen_base_tensors": missing,
        "training_scope": "language only",
    }
    workflow.write_json(export / "opd-export.json", result)
    (export / ".complete").touch()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    spec = json.loads((args.root / "run-spec.json").read_text())
    prepared = json.loads((args.root / "prepared.json").read_text())
    count = spec["training"]["num_rollouts"]
    result = {
        "updates": audit_training(args.root, count, spec["distillation"]["kl_coef"]),
        "export": complete_export(args.root, Path(prepared["model"]), count - 1),
    }
    workflow.write_json(args.root / "audit.json", result)


if __name__ == "__main__":
    main()
