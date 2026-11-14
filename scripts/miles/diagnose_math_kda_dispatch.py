"""Causal replay of captured application KDA inputs; never a qualification gate.

Replay each receiver's native autotuner choices on identical saved first-prefill
inputs. Only this isolated process's caches change, never kernel source or a live
server. The saved inputs come from the real Core publication diagnostic.
"""

import argparse
import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path

import torch
from scripts.miles import qualify_math_weight_publication as publication


def restore_dispatch(snapshot):
    """Install recorded native choices, requiring exact config membership."""
    installed = {}
    for path, entry in snapshot.items():
        if not entry["cache"]:
            continue
        pieces = path.split(".")
        module_end = next(index for index in range(len(pieces), 0, -1) if pieces[index - 1] != "fn")
        # Module names end before the exported kernel attribute. Nested .fn
        # entries represent Triton heuristic/autotuner wrappers.
        value = importlib.import_module(".".join(pieces[: module_end - 1]))
        for attribute in pieces[module_end - 1 :]:
            value = getattr(value, attribute)
        cache = {}
        for key, details in entry["cache"].items():
            matches = [
                config
                for config in value.configs
                if all(getattr(config, name, None) == setting for name, setting in details.items())
            ]
            if len(matches) != 1:
                raise ValueError(f"Recorded config not uniquely present in pinned runtime: {path}: {details}")
            cache[ast.literal_eval(key)] = matches[0]
        value.cache.clear()
        value.cache.update(cache)
        installed[path] = len(cache)
    return installed


def difference(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("Captured/replayed output layout differs")
    delta = (a.float() - b.float()).abs()
    return {
        "equal": torch.equal(a, b),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "max_abs": float(delta.max()),
        "changed_elements": int(torch.count_nonzero(delta)),
        "elements": a.numel(),
    }


def single_config_changes(base, target):
    """Vary one recorded native cache at a time, keeping all others fixed."""
    if base.keys() != target.keys():
        raise ValueError("Captured native autotuner sets differ")
    for path in base:
        if base[path]["cache"] != target[path]["cache"]:
            snapshot = copy.deepcopy(base)
            snapshot[path] = copy.deepcopy(target[path])
            yield path, snapshot


def unequal_paths(a, b, path="inputs"):
    """Identify every changed input/state without moving captures onto a GPU."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return [] if a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b) else [path]
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return [path]
        return [item for key in a for item in unequal_paths(a[key], b[key], f"{path}.{key}")]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [path]
        return [item for index, (x, y) in enumerate(zip(a, b)) for item in unequal_paths(x, y, f"{path}[{index}]")]
    return [] if a == b else [path]


def inspect_captures(records, captured, expected_index=None):
    """Gate causal replay on complete inputs and matching native semantics."""
    semantics = ("prefill_recurrence_index", "core_compat", "allow_neg_eigval")
    # Older first-recurrence captures did not record the index. Keep that
    # provenance unknown; an explicit expected index always requires a receipt.
    receipts = [{"prefill_recurrence_index": None, **record[1]} for record in records]
    for receipt in receipts:
        if any(key not in receipt for key in semantics):
            raise ValueError("Capture receipt lacks recurrence/native semantics")
        if expected_index is not None and receipt["prefill_recurrence_index"] != expected_index:
            raise ValueError("Captured recurrence index differs from requested diagnostic")
    pairs = []
    outputs = [torch.load(record[3], map_location="cpu", weights_only=True) for record in records]
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            changed = unequal_paths(captured[left], captured[right])
            changed_semantics = [key for key in semantics if receipts[left][key] != receipts[right][key]]
            pairs.append(
                {
                    "receivers": [records[left][0], records[right][0]],
                    "unequal_input_paths": changed,
                    "unequal_semantics": changed_semantics,
                    "causal_replay_allowed": not changed and not changed_semantics,
                    "output_comparison": difference(outputs[left], outputs[right]),
                    "unequal_dispatch_paths": unequal_paths(
                        receipts[left]["fla_dispatch"], receipts[right]["fla_dispatch"], "fla_dispatch"
                    ),
                }
            )
    return {
        "diagnostic_only": True,
        "qualification_passed": False,
        "prefill_recurrence_index": receipts[0]["prefill_recurrence_index"],
        "causal_replay_allowed": bool(pairs) and all(pair["causal_replay_allowed"] for pair in pairs),
        "pairs": pairs,
        "captures": [
            {
                "receiver": name,
                "input_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "receipt": receipt,
            }
            for name, receipt, inputs, output in records
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bisect-configs", action="store_true")
    parser.add_argument("--inspect-only", action="store_true", help="Write CPU input/state comparison without GPU use")
    parser.add_argument("--expected-recurrence-index", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for directory in sorted(args.capture.glob("gpu*")):
        (receipt_path,) = directory.glob("kda-receipt-*.json")
        (inputs_path,) = directory.glob("kda-inputs-*.pt")
        (output_path,) = directory.glob("kda-output-*.pt")
        records.append((directory.name, json.loads(receipt_path.read_text()), inputs_path, output_path))
    if len(records) != 3:
        raise ValueError("Require all three captured application receivers")
    captured = [torch.load(record[2], map_location="cpu", weights_only=True) for record in records]

    inspection = inspect_captures(records, captured, args.expected_recurrence_index)
    publication.save(args.output / "capture-inspection.json", inspection)
    if args.inspect_only:
        return
    if not inspection["causal_replay_allowed"]:
        raise ValueError("Captured receivers have unequal native KDA inputs/state")
    torch.cuda.set_device(args.gpu)
    importlib.import_module("olmo_sglang").register()
    backend = importlib.import_module("olmo_sglang.kda.backend")
    cpu = captured[0]
    device = torch.device("cuda", args.gpu)
    positional = [value.to(device) for value in cpu["positional"]]
    options = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in cpu["options"].items()
    }
    indices = options["cache_indices"]
    initial = cpu["initial_state"].to(device)
    states = torch.zeros((int(indices.max()) + 1, *initial.shape[1:]), dtype=initial.dtype, device=device)
    if not bool(cpu["valid"].all()):
        raise ValueError("Diagnostic requires the captured all-valid first prefill")
    kernel = backend.OlmoFLAKDAKernel(allow_neg_eigval=records[0][1]["allow_neg_eigval"])
    kernel.core_compat = records[0][1]["core_compat"]
    outputs = []
    checks = []
    for name, receipt, inputs_path, output_path in records:
        if (receipt["core_compat"], receipt["allow_neg_eigval"]) != (kernel.core_compat, kernel.allow_neg_eigval):
            raise ValueError("Native dispatch semantics differ across captures")
        installed = restore_dispatch(receipt["fla_dispatch"])
        repeats = []
        for repeat in range(2):
            states.zero_()
            states.index_copy_(0, indices.long(), initial)
            with torch.inference_mode():
                result = kernel._run(*positional, **options, ssm_states=states)
            torch.cuda.synchronize()
            result = result.cpu()
            torch.save(result, args.output / f"{name}-repeat{repeat}.pt")
            repeats.append(result)
        expected = torch.load(output_path, map_location="cpu", weights_only=True)
        outputs.append(repeats[0])
        checks.append(
            {
                "receiver_config": name,
                "installed": installed,
                "captured_output_agreement": difference(repeats[0], expected),
                "repeat_agreement": difference(*repeats),
                "input_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
                "actual_dispatch": publication.fla_dispatch_snapshot(),
            }
        )
        publication.save(args.output / "progress.json", checks)
    interventions = []
    if args.bisect_configs:
        base = records[0][1]["fla_dispatch"]
        for target_index, (name, receipt, _, _) in enumerate(records[1:], 1):
            for path, snapshot in single_config_changes(base, receipt["fla_dispatch"]):
                installed = restore_dispatch(snapshot)
                repeats = []
                for _ in range(2):
                    states.zero_()
                    states.index_copy_(0, indices.long(), initial)
                    with torch.inference_mode():
                        result = kernel._run(*positional, **options, ssm_states=states)
                    torch.cuda.synchronize()
                    repeats.append(result.cpu())
                interventions.append(
                    {
                        "target_receiver": name,
                        "changed_autotuner": path,
                        "installed": installed,
                        "base_output_comparison": difference(repeats[0], outputs[0]),
                        "target_output_comparison": difference(repeats[0], outputs[target_index]),
                        "repeat_agreement": difference(*repeats),
                        "actual_dispatch": publication.fla_dispatch_snapshot(),
                    }
                )
                publication.save(args.output / "config-interventions.json", interventions)
    publication.save(
        args.output / "diagnostic-complete.json",
        {
            "diagnostic_only": True,
            "qualification_passed": False,
            "kernel_source_changed": False,
            "runtime_dependencies_changed": False,
            "live_server_modified": False,
            "capture": str(args.capture),
            "gpu": args.gpu,
            "identical_inputs_state": True,
            "checks": checks,
            "single_config_interventions": interventions,
            "config_pair_differences": [
                {"a": records[0][0], "b": records[index][0], **difference(outputs[0], outputs[index])}
                for index in (1, 2)
            ],
        },
    )


if __name__ == "__main__":
    main()
