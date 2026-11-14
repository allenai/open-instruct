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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bisect-configs", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(args.gpu)
    importlib.import_module("olmo_sglang").register()
    backend = importlib.import_module("olmo_sglang.kda.backend")
    records = []
    for directory in sorted(args.capture.glob("gpu*")):
        (receipt_path,) = directory.glob("kda-receipt-*.json")
        (inputs_path,) = directory.glob("kda-inputs-*.pt")
        (output_path,) = directory.glob("kda-output-*.pt")
        records.append((directory.name, json.loads(receipt_path.read_text()), inputs_path, output_path))
    if len(records) != 3:
        raise ValueError("Require all three captured application receivers")
    captured = [torch.load(record[2], map_location="cpu", weights_only=True) for record in records]

    # The causal comparison must hold EVERY input and state constant.
    def compare_tree(a, b):
        if isinstance(a, torch.Tensor):
            return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(compare_tree(a[key], b[key]) for key in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(compare_tree(x, y) for x, y in zip(a, b))
        return a == b

    if not all(compare_tree(captured[0], other) for other in captured[1:]):
        raise ValueError("Captured receivers have unequal native KDA inputs/state")
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
