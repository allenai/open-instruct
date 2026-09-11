"""Compare retained update-zero prefill observations; never infer decode equivalence."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

BACKENDS = ("core", "megatron")
PHASES = ("hf", "published")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_difference(left, right, positions):
    require(isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor), "Expected captured tensors")
    require(
        left.shape == right.shape and left.ndim >= 1 and left.shape[0] == len(positions), "Tensor row shapes differ"
    )
    a, b = left.double().reshape(len(positions), -1), right.double().reshape(len(positions), -1)
    require(torch.isfinite(a).all() and torch.isfinite(b).all(), "Nonfinite captured tensor")
    delta = (a - b).abs()
    row_l2 = torch.linalg.vector_norm(a - b, dim=1)
    denominator = torch.linalg.vector_norm(a, dim=1)
    relative = row_l2 / denominator.clamp_min(1e-30)
    return {
        "shape": list(left.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "dtype_equal": left.dtype == right.dtype,
        "exact_values": bool(torch.equal(a, b)),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "relative_l2": float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a).clamp_min(1e-30)),
        "rows": {
            "positions": positions,
            "max_abs": delta.max(1).values.tolist(),
            "mean_abs": delta.mean(1).tolist(),
            "relative_l2": relative.tolist(),
            "reference_norm": denominator.tolist(),
            "exact": (delta.max(1).values == 0).tolist(),
        },
    }


def route_difference(left, right):
    a, b = left["topk_ids"].long(), right["topk_ids"].long()
    logits_a, logits_b = left["logits"], right["logits"]
    require(a.shape == b.shape and a.ndim == 2 and a.shape[0] == logits_a.shape[0], "Route row shapes differ")
    for ids, logits in ((a, logits_a), (b, logits_b)):
        require(bool(((ids >= 0) & (ids < logits.shape[-1])).all()), "Expert IDs outside router vocabulary")
        require(
            bool((ids.sort(-1).values[:, 1:] != ids.sort(-1).values[:, :-1]).all()), "Duplicate expert in selected set"
        )
    positions = list(range(a.shape[0]))
    logits = tensor_difference(logits_a, logits_b, positions)
    overlap = (a[:, :, None] == b[:, None, :]).any(-1).sum(-1)
    same_set, same_order = overlap == a.shape[1], (a == b).all(-1)
    margin_a = logits_a.float().sort(dim=-1, descending=True).values
    margin_b = logits_b.float().sort(dim=-1, descending=True).values
    k = a.shape[1]
    require(k < logits_a.shape[-1], "Router has no k+1 boundary")
    margin_a = margin_a[:, k - 1] - margin_a[:, k]
    margin_b = margin_b[:, k - 1] - margin_b[:, k]
    max_delta = torch.tensor(logits["rows"]["max_abs"], dtype=torch.float64)
    separated = margin_a.double() > 2 * max_delta
    violations = separated & ~same_set
    return {
        "tokens": a.shape[0],
        "top_k": k,
        "assignment_agreement": float(overlap.double().mean() / k),
        "exact_set_fraction": float(same_set.double().mean()),
        "exact_order_fraction": float(same_order.double().mean()),
        "changed_set_positions": (~same_set).nonzero().flatten().tolist(),
        "changed_order_positions": (~same_order).nonzero().flatten().tolist(),
        "expert_ids_left": a.tolist(),
        "expert_ids_right": b.tolist(),
        "overlap_per_token": overlap.tolist(),
        "left_boundary_margin": margin_a.tolist(),
        "right_boundary_margin": margin_b.tolist(),
        "margin_exceeds_twice_delta": separated.tolist(),
        "stability_bound_violation_positions": violations.nonzero().flatten().tolist(),
        "left_canonical_set_matches": left["canonical_set_matches"].tolist(),
        "right_canonical_set_matches": right["canonical_set_matches"].tolist(),
        "logits": logits,
        "weights": tensor_difference(left["topk_weights"], right["topk_weights"], positions),
        "weight_comparison_note": "Weights compared in emitted slot order; use expert IDs to distinguish reordering from a changed set.",
    }


def stage_key(name):
    if not name.startswith("model.layers."):
        return (
            -1 if name != "model.norm" else 100000,
            {"model.embed_tokens": 0, "model.embed_norm": 1}.get(name, 0),
            name,
        )
    layer = int(name.split(".")[2])
    suffix = ".".join(name.split(".")[3:])
    stages = {
        "input": 0,
        "pre_attention_layernorm": 1,
        "self_attn": 2,
        "post_attention_layernorm": 3,
        "pre_feedforward_layernorm": 4,
        "mlp.input": 5,
        "mlp.latent_down_proj": 6,
        "mlp.latent_up_proj": 7,
        "mlp": 8,
        "post_feedforward_layernorm": 9,
        "": 10,
    }
    return layer, stages.get(suffix, 11), name


def compare_captures(left, right):
    require(left["input_ids"] == right["input_ids"], "Teacher-forced input IDs differ")
    require(left["positions"] == right["positions"], "Activation token positions differ")
    require(set(left["activations"]) == set(right["activations"]), "Activation inventory differs")
    require(set(left["routes"]) == set(right["routes"]), "Route inventory differs")
    activations = {
        name: tensor_difference(left["activations"][name], right["activations"][name], left["positions"])
        for name in sorted(left["activations"], key=stage_key)
    }
    routes = {
        name: route_difference(left["routes"][name], right["routes"][name])
        for name in sorted(left["routes"], key=stage_key)
    }
    sources_a, sources_b = left["sources"], right["sources"]
    changed_sources = {
        name: {"left": sources_a.get(name), "right": sources_b.get(name)}
        for name in sorted(sources_a.keys() | sources_b.keys())
        if sources_a.get(name, {}).get("sha256") != sources_b.get(name, {}).get("sha256")
    }
    control_keys = (left["controls"].keys() | right["controls"].keys()) - {"pid"}
    controls = {
        name: {"left": left["controls"].get(name), "right": right["controls"].get(name)}
        for name in sorted(control_keys)
        if left["controls"].get(name) != right["controls"].get(name)
    }
    parameter_dtypes = {
        name: {"left": left["parameter_dtypes"].get(name), "right": right["parameter_dtypes"].get(name)}
        for name in sorted(left["parameter_dtypes"].keys() | right["parameter_dtypes"].keys())
        if left["parameter_dtypes"].get(name) != right["parameter_dtypes"].get(name)
    }
    result = {
        "left": {k: left[k] for k in ("phase", "case_id", "capture_id")},
        "right": {k: right[k] for k in ("phase", "case_id", "capture_id")},
        "input_tokens": len(left["input_ids"]),
        "activation_positions": left["positions"],
        "earliest_nonexact_activation": next(
            (name for name, diff in activations.items() if not diff["exact_values"]), None
        ),
        "earliest_changed_route_set": next(
            (name for name, diff in routes.items() if diff["changed_set_positions"]), None
        ),
        "all_activation_values_exact": all(diff["exact_values"] for diff in activations.values()),
        "all_route_sets_exact": all(not diff["changed_set_positions"] for diff in routes.values()),
        "activations": activations,
        "routes": routes,
        "changed_source_hashes": changed_sources,
        "changed_execution_controls": controls,
        "changed_parameter_dtype_or_shape": parameter_dtypes,
    }
    available = "autotune_configs" in left and "autotune_configs" in right
    result["autotune_observations"] = {"available_both": available}
    if available:
        choices_a, choices_b = left["autotune_configs"], right["autotune_configs"]
        result["autotune_observations"].update(
            {
                "populated_cache_counts": {"left": len(choices_a), "right": len(choices_b)},
                "changed_configs": {
                    name: {"left": choices_a.get(name), "right": choices_b.get(name)}
                    for name in sorted(choices_a.keys() | choices_b.keys())
                    if choices_a.get(name) != choices_b.get(name)
                },
                "left_policy": left.get("autotune_policy"),
                "right_policy": right.get("autotune_policy"),
                "interpretation": "Populated caches include warmup/history; equal or empty caches do not prove equal kernels for every launch.",
            }
        )
    require(("next_token_logits" in left) == ("next_token_logits" in right), "Next-token logit capture differs")
    if "next_token_logits" in left:
        a, b = left["next_token_logits"], right["next_token_logits"]
        result["next_token_logits"] = tensor_difference(a, b, list(range(a.shape[0])))
        result["next_token_argmax"] = {"left": a.argmax(-1).tolist(), "right": b.argmax(-1).tolist()}
    return result


def load_capture(directory, capture_id, expected_ids):
    files = list(directory.glob(f"trace/worker-*/{capture_id}.json"))
    require(len(files) == 1, f"{capture_id}: expected exactly one worker capture")
    metadata = json.loads(files[0].read_text())
    path = files[0].with_suffix(".pt")
    require(sha256(path) == metadata["sha256"], f"{capture_id}: tensor file hash differs")
    data = torch.load(path, map_location="cpu", weights_only=True)
    require(
        data["capture_id"] == capture_id and data["input_ids"] == expected_ids,
        f"{capture_id}: identity/input mismatch",
    )
    expected_phase, expected_case = capture_id.rsplit("-", 1)[0].split("-", 1)
    require(data["phase"] == expected_phase and data["case_id"] == expected_case, f"{capture_id}: phase/case mismatch")
    require(metadata["positions"] == data["positions"], f"{capture_id}: metadata positions differ")
    require(
        data["sources"] == metadata["sources"] and data["controls"] == metadata["controls"],
        f"{capture_id}: source/control metadata differs",
    )
    for field in ("autotune_configs", "autotune_policy"):
        require(data.get(field) == metadata.get(field), f"{capture_id}: autotune metadata differs")
    expected_positions = sorted(
        set(range(min(16, len(expected_ids)))) | set(range(max(0, len(expected_ids) - 128), len(expected_ids)))
    )
    require(data["positions"] == expected_positions, f"{capture_id}: unexpected activation row selection")
    return data


def response_control(directory, phase, case_id):
    responses = {
        kind: json.loads((directory / f"{phase}-{case_id}-{kind}-response.json").read_text())
        for kind in ("control", "capture")
    }

    def stable(response):
        meta = response.get("meta_info", {})
        return {
            "text": response.get("text"),
            "output_ids": response.get("output_ids"),
            "logprobs": {key: value for key, value in meta.items() if "logprob" in key},
        }

    a, b = stable(responses["control"]), stable(responses["capture"])
    return {
        "phase": phase,
        "case_id": case_id,
        "same_output_text": a["text"] == b["text"],
        "same_output_ids": a["output_ids"] == b["output_ids"]
        if a["output_ids"] is not None and b["output_ids"] is not None
        else None,
        "exact_logprob_fields": a["logprobs"] == b["logprobs"] if a["logprobs"] and b["logprobs"] else None,
        "control": a,
        "capture": b,
    }


def compare_campaign(root, *, core_root=None, megatron_root=None, hf_only=False, evidence_only=False):
    require(not (hf_only and evidence_only), "Choose one limited evidence mode")
    root = Path(root)
    directories = {
        "core": Path(core_root) if core_root else root / "core",
        "megatron": Path(megatron_root) if megatron_root else root / "megatron",
    }
    manifests = {backend: json.loads((directories[backend] / "manifest.json").read_text()) for backend in BACKENDS}
    cases = manifests["core"]["inputs"]["cases"]
    require(cases == manifests["megatron"]["inputs"]["cases"], "Frozen input cases differ between backends")
    require(
        len(cases) == 4 and len({case["case_id"] for case in cases}) == 4, "Expected four unique fixed-prefix cases"
    )
    phases = ("hf",) if hf_only else PHASES
    cleanup_results = {}
    for backend in () if hf_only else BACKENDS:
        complete = json.loads((directories[backend] / "probe-complete.json").read_text())
        cleanup = json.loads((directories[backend] / "cleanup.json").read_text())
        require(
            complete["completed"] and complete["optimizer_calls"] == 0 and complete["initial_publications"] == 1,
            f"{backend}: incomplete zero-update protocol",
        )
        cleanup_results[backend] = cleanup
        if not evidence_only:
            require(cleanup["completed"], f"{backend}: cleanup failed")
    reports, observers = [], []
    # Load one case at a time to keep CPU memory bounded below full-model size.
    for case in cases:
        case_id = case["case_id"]
        loaded = {
            (backend, phase): load_capture(directories[backend], f"{phase}-{case_id}-capture", case["input_ids"])
            for backend in BACKENDS
            for phase in phases
        }
        for phase in phases:
            reports.append(
                {
                    "comparison": "cross_backend",
                    "case_id": case_id,
                    "phase": phase,
                    **compare_captures(loaded["core", phase], loaded["megatron", phase]),
                }
            )
        for backend in BACKENDS:
            if not hf_only:
                reports.append(
                    {
                        "comparison": "before_after_publication",
                        "backend": backend,
                        "case_id": case_id,
                        **compare_captures(loaded[backend, "hf"], loaded[backend, "published"]),
                    }
                )
            for phase in phases:
                observers.append({"backend": backend, **response_control(directories[backend], phase, case_id)})
                if case is cases[0]:
                    repeated = load_capture(directories[backend], f"{phase}-{case_id}-repeat", case["input_ids"])
                    reports.append(
                        {
                            "comparison": "repeated_prefix",
                            "backend": backend,
                            "case_id": case_id,
                            "phase": phase,
                            **compare_captures(loaded[backend, phase], repeated),
                        }
                    )
    report = {
        "schema_version": 1,
        "valid": True,
        "validity_scope": "HF prefill only; full protocol incomplete"
        if hf_only
        else "Capture evidence only; cleanup failures retained and no clean-protocol success asserted"
        if evidence_only
        else "Complete verified observations, not a numerical-equivalence verdict",
        "full_protocol_complete": not hf_only and not evidence_only,
        "captured_phases_complete": not hf_only,
        "evidence_only": evidence_only,
        "cleanup_results": cleanup_results,
        "backend_directories": {name: str(path) for name, path in directories.items()},
        "interpretation": "Observed fixed-prefix prefill comparisons. Exact equality here does not establish equivalence of historical autoregressive decode, batching, CUDA graph execution, or training. Tracing introduces CPU synchronization; unarmed controls and repeated prefixes quantify only the observed cases.",
        "manifests": manifests,
        "comparisons": reports,
        "observer_controls": observers,
        "protocol_files": {
            backend: {
                name: json.loads((directories[backend] / name).read_text())
                for name in (
                    "resolved-arguments.json",
                    "serving-resolved.json",
                    "serving-topology.json",
                    "initial-weight-comparison.json",
                    "probe-complete.json",
                    "cleanup.json",
                )
                if not hf_only or (directories[backend] / name).is_file()
            }
            for backend in BACKENDS
        },
    }

    report["participants"] = {
        side: {
            "backend": manifests[backend].get("backend", backend),
            "image": manifests[backend].get("image"),
            "campaign": directories[backend].parent.name,
            "directory": str(directories[backend]),
        }
        for side, backend in zip(("left", "right"), BACKENDS, strict=True)
    }
    same_backend = manifests["core"].get("backend") == manifests["megatron"].get("backend")
    if same_backend and manifests["core"].get("backend") is not None:
        slots = dict(zip(BACKENDS, ("left", "right"), strict=True))
        for field in ("backend_directories", "manifests", "cleanup_results", "protocol_files"):
            report[field] = {slots[key]: value for key, value in report[field].items()}
        for row in reports + observers:
            if row.get("comparison") == "cross_backend":
                row["comparison"] = "cross_run"
            if "backend" in row:
                row["run"] = slots[row.pop("backend")]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, help="Separate Core retry directory")
    parser.add_argument("--megatron-root", type=Path, help="Separate original Megatron directory")
    parser.add_argument(
        "--hf-only",
        action="store_true",
        help="Strict limited scope: four HF cases and repeats; no full-protocol claim",
    )
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="Require all captured phases but retain cleanup failures without claiming protocol success",
    )
    args = parser.parse_args()
    report = compare_campaign(
        args.root,
        core_root=args.core_root,
        megatron_root=args.megatron_root,
        hf_only=args.hf_only,
        evidence_only=args.evidence_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"valid": True, "comparisons": len(report["comparisons"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
