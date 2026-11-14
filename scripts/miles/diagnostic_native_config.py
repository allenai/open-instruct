"""Recorded native-cache intervention for isolated publication diagnostics only."""

import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path

TARGET = "fla.ops.kda.chunk_intra_token_parallel.chunk_kda_fwd_kernel_intra_token_parallel.fn"
GATE_TARGET = "fla.ops.kda.gate.kda_gate_chunk_cumsum_vector_kernel.fn"


def runtime_key(tuner, args, kwargs):
    """Mirror the pinned Triton Autotuner.run key; source hash checked on install."""
    all_args = {**dict(zip(tuner.arg_names, args)), **kwargs}
    selected = {name: value for name, value in all_args.items() if name in tuner.arg_names}
    key = [selected[name] for name in tuner.keys if name in selected]
    key.extend(str(value.dtype) for value in selected.values() if hasattr(value, "dtype"))
    return tuple(key)


def install(receipt_path, output, save):
    """Control recorded causal keys in isolated servers; validate all before mutation."""
    receipt = json.loads(Path(receipt_path).read_text())
    extra = receipt.get("additional_interventions", [])
    if len(extra) > 1 or (extra and extra[0].get("target") != GATE_TARGET):
        raise ValueError("Only the recorded residual gate intervention may be added")
    if receipt.get("target") != TARGET:
        raise ValueError("Primary intervention must be the recorded intra-token tuner")
    receipt_hash = hashlib.sha256(Path(receipt_path).read_bytes()).hexdigest()
    installs = [_prepare(receipt, receipt_hash, output, save)]
    for entry in extra:
        installs.append(_prepare(entry, receipt_hash, output / "gate-native-intervention", save))
    for apply in installs:
        apply()


def _prepare(receipt, receipt_hash, output, save):
    """Resolve one key/config without changing a tuner until every guard passes."""
    target = receipt["target"]
    if target not in (TARGET, GATE_TARGET) or receipt.get("diagnostic_only") is not True:
        raise ValueError("Only the approved isolated native tuner intervention is allowed")
    module, attribute, wrapper = target.rsplit(".", 2)
    tuner = getattr(getattr(importlib.import_module(module), attribute), wrapper)
    original = tuner.run
    source_hash = hashlib.sha256(inspect.getsource(type(tuner).run).encode()).hexdigest()
    if receipt["autotuner_run_sha256"] != source_hash:
        raise ValueError("Pinned autotuner key implementation changed")
    if len(receipt["cache"]) != 1:
        raise ValueError("Require exactly one recorded native key")
    key_text, details = next(iter(receipt["cache"].items()))
    key = ast.literal_eval(key_text)
    matches = [c for c in tuner.configs if all(getattr(c, k, None) == v for k, v in details.items())]
    if len(matches) != 1:
        raise ValueError("Recorded native configuration not uniquely present in runtime")
    config = matches[0]
    record = {
        "diagnostic_only": True,
        "qualification_passed": False,
        "receipt_sha256": receipt_hash,
        "target": target,
        "key": key_text,
        "config": details,
        "runtime_membership_verified": True,
        "autotuner_run_sha256": source_hash,
        "matched_calls": 0,
        "other_key_calls": 0,
    }

    def run(*args, **kwargs):
        actual_key = runtime_key(tuner, args, kwargs)
        matched = actual_key == key
        if matched:
            tuner.cache[key] = config
            save(output / "native-intervention-pre.json", {**record, "actual_key": repr(actual_key)})
        result = original(*args, **kwargs)
        if matched:
            if tuner.cache.get(key) is not config or tuner.best_config is not config:
                raise ValueError("Native launch did not retain recorded configuration")
            record["matched_calls"] += 1
        else:
            record["other_key_calls"] += 1
        save(output / "native-intervention-post.json", {**record, "actual_key": repr(actual_key)})
        return result

    def apply():
        output.mkdir(parents=True, exist_ok=True)
        save(output / "native-intervention-installed.json", record)
        tuner.run = run

    return apply
