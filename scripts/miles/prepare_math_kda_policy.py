"""Freeze candidate native dispatch membership; this does not qualify numerics.

Run inside the exact pinned runtime. The supplied candidate has the same
native target/cache fields as the retained causal receipts. It is evidence for
selection, not a diagnostic hook used by the serving policy.
"""

import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path

from open_instruct.miles.execution import kda_dispatch


def prepare(candidate, *, precision, kv_cache_dtype, model):
    if precision not in ("bf16", "nvfp4") or kv_cache_dtype not in ("bf16", "nvfp4"):
        raise ValueError("Unsupported study precision")
    if precision == "bf16" and kv_cache_dtype != "bf16":
        raise ValueError("BF16 arm retains BF16 KV")
    data = json.loads(candidate.read_text())
    entries = []
    for source in [data, *data.get("additional_interventions", [])]:
        target = source["target"]
        if target not in (*kda_dispatch.TARGETS, *kda_dispatch.OPTIONAL_TARGETS):
            raise ValueError("Unsupported native target")
        module_name, attribute, wrapper = target.rsplit(".", 2)
        module = importlib.import_module(module_name)
        tuner = getattr(getattr(module, attribute), wrapper)
        entry = {
            "target": target,
            "module_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            "run_sha256": hashlib.sha256(inspect.getsource(type(tuner).run).encode()).hexdigest(),
            "key_names": list(tuner.keys),
            "configs": source["cache"],
        }
        # Use the same exact source/schema/membership validation as serving;
        # discard the callable, leaving all native caches and dispatch intact.
        if entry["run_sha256"] != source["autotuner_run_sha256"]:
            raise ValueError("Candidate autotuner source differs from installed runtime")
        kda_dispatch.prepare(tuner, entry, "preflight-only")
        entries.append(entry)
    kda_dispatch.validate_targets(entries)
    scope = {
        "quantization": "nvfp4_online" if precision == "nvfp4" else None,
        "kv_cache_dtype": kv_cache_dtype,
        "enable_fp32_lm_head": True,
        "attention_backend": "trtllm_mha" if kv_cache_dtype == "nvfp4" else "triton",
    }
    if kv_cache_dtype == "nvfp4":
        scope.update(page_size=16, prefill_kv_cache_dequant_dtype="nvfp4")
    return {
        "version": 1,
        "device_capability": [10, 0],
        "serving_scope": scope,
        "entries": entries,
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "runtime_lock_sha256": hashlib.sha256(Path("/opt/core-rl/sources/runtime.lock.json").read_bytes()).hexdigest(),
        "model_config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
        "precision": precision,
        "numerical_qualification": False,
        "scope": "Exact native key candidates only. New short/full-cap repeated publication gates required; unknown keys fail.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "nvfp4"), required=True)
    parser.add_argument("--kv-cache-dtype", choices=("bf16", "nvfp4"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.candidate, precision=args.precision, kv_cache_dtype=args.kv_cache_dtype, model=args.model)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
