"""Audit the tiny sampled-token update and complete its HF model export."""

import argparse
import ast
import json
import math
import re
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
        # With distillation.use_rollout_logprobs the trainer never scores the student, so the
        # dump only carries the rollout engine's log-probs; that is the tensor the advantages used.
        student_key = "log_probs" if "log_probs" in data else "rollout_log_probs"
        student, teacher, advantages = (data[key] for key in (student_key, "teacher_log_probs", "advantages"))
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
                "student_log_probs": student_key,
                "samples": len(student),
                "max_advantage_error": max(errors),
                "max_abs_opd_signal": max(signals),
            }
        )
    return records


def audit_optimizer(root, num_rollouts, optimizer_steps_per_rollout=1):
    steps = {}
    for match in re.finditer(r"step \d+: (\{[^\n]+\})", (root / "training.log").read_text()):
        values = ast.literal_eval(match[1])
        if "train/grad_norm" in values:
            steps[int(values["train/step"])] = values
    if sorted(steps) != list(range(num_rollouts * optimizer_steps_per_rollout)):
        raise ValueError("Missing optimizer step metrics")
    for values in steps.values():
        if not all(math.isfinite(v) for v in values.values() if isinstance(v, (int, float))):
            raise ValueError("Nonfinite optimizer metric")
        if values["train/grad_norm"] <= 0:
            raise ValueError("OPD update has zero gradient norm")
    return list(steps.values())


def complete_export(root, base, rollout_id):
    """Preserve frozen vision/MTP weights omitted by the native language-only exporter."""
    export = root / f"hf-{rollout_id}"
    if not (export / ".complete").exists():
        raise ValueError(f"Native HF export did not complete: {export}")
    index_path = export / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    original = json.loads((base / "model.safetensors.index.json").read_text())
    original_map, weight_map = original["weight_map"], index["weight_map"]
    changed, fp32_a_logs, changed_since_previous = 0, 0, 0
    # Exports land every save interval, so the previous export is the newest hf-N below this one.
    previous_ids = sorted(
        int(path.name[3:]) for path in root.glob("hf-*") if path.name[3:].isdigit() and int(path.name[3:]) < rollout_id
    )
    previous = root / f"hf-{previous_ids[-1]}" if previous_ids else None
    previous_map = (
        json.loads((previous / "model.safetensors.index.json").read_text())["weight_map"] if previous else {}
    )
    for shard in sorted(set(weight_map.values())):
        tensors = safetensors_torch.load_file(export / shard)
        for name, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Export contains nonfinite weights: {name}")
            if name.endswith("A_log"):
                if tensor.dtype != torch.float32:
                    raise ValueError(f"Export lost A_log precision: {name}")
                fp32_a_logs += 1
            if name in previous_map:
                with safe_open(previous / previous_map[name], framework="pt", device="cpu") as source:
                    changed_since_previous += int(not torch.equal(source.get_tensor(name), tensor))
            if name in original_map:
                with safe_open(base / original_map[name], framework="pt", device="cpu") as source:
                    old = source.get_tensor(name)
                changed += int(not torch.equal(old, tensor))
    if changed == 0 or fp32_a_logs == 0:
        raise ValueError("Export lacks changed weights or FP32 A_log tensors")
    if previous is not None and not changed_since_previous:
        raise ValueError("Weights did not change between learner updates")
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
        "previous_export": str(previous) if previous else None,
        "changed_since_previous_update": changed_since_previous,
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
        "optimizer": audit_optimizer(args.root, count, spec["training"].get("optimizer_steps_per_rollout", 1)),
        "updates": audit_training(args.root, count, spec["distillation"]["kl_coef"]),
        "export": complete_export(args.root, Path(prepared["model"]), count - 1),
    }
    workflow.write_json(args.root / "audit.json", result)


if __name__ == "__main__":
    main()
