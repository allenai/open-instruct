"""Audit the tiny sampled-token update and complete its HF model export."""

import argparse
import ast
import dataclasses
import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors import torch as safetensors_torch

from open_instruct.miles import eopd_math, workflow
from open_instruct.miles.opd_prepare import _INDEX_NAME, _VL_TEXT_PREFIX


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


def audit_eopd(root, num_rollouts, settings, optimizer_steps):
    """Check the teacher top-k that reached the trainer and the gated forward-KL metrics it logged.

    The train-data dump holds each sample's ``metadata`` (top-k ids/log-probs, ``[R, k]``); the
    gate is re-derived from those log-probs. The FKL itself needs the student's logits, so its
    evidence is the per-step ``train/eopd_*`` metrics: finite, gate fraction in [0, 1] and a
    nonnegative FKL (zero only if no token was gated).
    """
    records = []
    for rollout_id in range(num_rollouts):
        data = torch.load(root / f"debug/train_data/{rollout_id}_0.pt", map_location="cpu", weights_only=False)
        data = data["rollout_data"]
        metadata, teacher = data.get("metadata"), data["teacher_log_probs"]
        if not metadata or len(metadata) != len(teacher):
            raise ValueError("EOPD training data lacks per-sample teacher top-k metadata")
        gates, entropies, masses, tokens = [], [], [], 0
        for sample_metadata, sampled in zip(metadata, teacher, strict=True):
            ids, log_probs = eopd_math.sample_tensors(sample_metadata, settings.top_k)
            if ids.shape[0] != sampled.shape[0]:
                raise ValueError("EOPD top-k covers different positions than the teacher scores")
            if not torch.isfinite(log_probs).all() or (log_probs > 1e-6).any():
                raise ValueError("EOPD teacher top-k log-probs are nonfinite or positive")
            if (ids.sort(dim=-1).values[:, 1:] == ids.sort(dim=-1).values[:, :-1]).any():
                raise ValueError("EOPD teacher top-k repeats a token id")
            gates.append(eopd_math.gate(log_probs, settings.tau))
            entropies.append(eopd_math.proxy_entropy(log_probs))
            masses.append(eopd_math.topk_mass(log_probs))
            tokens += ids.shape[0]
        records.append(
            {
                "rollout_id": rollout_id,
                "samples": len(metadata),
                "tokens": tokens,
                "gate_frac": float(torch.cat(gates).mean()),
                "proxy_entropy_mean": float(torch.cat(entropies).mean()),
                "topk_mass_mean": float(torch.cat(masses).mean()),
            }
        )
    for values in optimizer_steps:
        for key in ("train/eopd_fkl_loss", "train/eopd_fkl", "train/eopd_gate_frac"):
            if key not in values or not math.isfinite(values[key]):
                raise ValueError(f"Missing or nonfinite {key} in the optimizer metrics")
        if not 0.0 <= values["train/eopd_gate_frac"] <= 1.0 or values["train/eopd_fkl_loss"] < 0:
            raise ValueError("EOPD metrics out of range")
        if values["train/eopd_gate_frac"] > 0 and values["train/eopd_fkl_loss"] == 0:
            raise ValueError("Gated tokens without a forward-KL contribution")
    return {"settings": dataclasses.asdict(settings), "rollouts": records}


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


def _weight_map(directory):
    """Tensor name -> shard file from the HF index, or from the shards of a single-file checkpoint."""
    index_path = directory / _INDEX_NAME
    if index_path.is_file():
        return json.loads(index_path.read_text())["weight_map"]
    weight_map = {}
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as source:
            weight_map.update(dict.fromkeys(source.keys(), shard.name))
    return weight_map


def complete_export(root, base, rollout_id):
    """Check the native HF export against the base and preserve frozen weights it omitted.

    Qwen3.5 exports are language-only, so the base's vision/MTP tensors are copied into a
    ``frozen-base.safetensors`` shard and the FP32 ``A_log`` tensors must survive; plain language
    models (Qwen3) have neither, and every base tensor must be present.
    """
    export = root / f"hf-{rollout_id}"
    if not (export / ".complete").exists():
        raise ValueError(f"Native HF export did not complete: {export}")
    index_path = export / _INDEX_NAME
    weight_map = _weight_map(export)
    index = (
        json.loads(index_path.read_text())
        if index_path.is_file()
        else {"metadata": {"total_size": 0}, "weight_map": weight_map}
    )
    original_map = _weight_map(base)
    multimodal_base = any(name.startswith(_VL_TEXT_PREFIX) for name in original_map)
    has_a_log = any(name.endswith("A_log") for name in original_map)
    changed, fp32_a_logs, changed_since_previous = 0, 0, 0
    # Exports land every save interval, so the previous export is the newest hf-N below this one.
    previous_ids = sorted(
        int(path.name[3:]) for path in root.glob("hf-*") if path.name[3:].isdigit() and int(path.name[3:]) < rollout_id
    )
    previous = root / f"hf-{previous_ids[-1]}" if previous_ids else None
    previous_map = _weight_map(previous) if previous else {}
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
    if changed == 0:
        raise ValueError("Export lacks changed weights")
    if has_a_log and fp32_a_logs == 0:
        raise ValueError("Export lost its FP32 A_log tensors")
    if previous is not None and not changed_since_previous:
        raise ValueError("Weights did not change between learner updates")
    missing = sorted(set(original_map) - set(weight_map))
    # Missing language weights would be an export failure, not frozen extras; a plain language
    # model has no frozen extras at all.
    if any(name.startswith(_VL_TEXT_PREFIX) or not multimodal_base for name in missing):
        raise ValueError(f"Native export omitted language model weights: {missing[:5]}")
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
        "training_scope": "language only" if multimodal_base else "full model",
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
    optimizer = audit_optimizer(args.root, count, spec["training"].get("optimizer_steps_per_rollout", 1))
    result = {"optimizer": optimizer, "updates": audit_training(args.root, count, spec["distillation"]["kl_coef"])}
    settings = eopd_math.Settings.from_distillation(spec["distillation"])
    if settings.enabled:
        result["eopd"] = audit_eopd(args.root, count, settings, optimizer)
    result["export"] = complete_export(args.root, Path(prepared["model"]), count - 1)
    workflow.write_json(args.root / "audit.json", result)


if __name__ == "__main__":
    main()
