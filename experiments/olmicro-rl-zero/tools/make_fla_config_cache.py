"""Build an FLA config-cache directory (FLA_CONFIG_DIR) from TRITON_PRINT_AUTOTUNING=1 logs.

fla-core 0.5.2 kernels decorated with `fla_cache_autotune` look up `$FLA_CONFIG_DIR/{kernel}.json` before
autotuning (`fla/ops/utils/cache.py`). With FLA_CACHE_MODE=default every call uses that file's `default_config`, so
the pack-length-keyed kernels (causal_conv1d fwd/bwd key ['D','W','NB'], l2norm fwd/bwd key ['D','NB'];
NB = cdiv(B*T, 1024) or cdiv(T, 65536)) stop re-autotuning for every new pack length. Kernels without a file, and
kernels using plain `triton.autotune` (e.g. layer_norm_gated_*), autotune as before.

For each kernel the default is the winner at the largest observed NB (the longest packs dominate training compute).
Every observed key is also stored under `autotune_entries` (exact matches, for FLA_CACHE_MODE=strict/full).

Usage: python make_fla_config_cache.py OUT_DIR LOG [LOG ...] [--kernels causal_conv1d_fwd_kernel ...]
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

# fla_cache_autotune kernels whose autotune key includes a pack-length bucket (NB) on the Olmicro GDN path.
DEFAULT_KERNELS = ("causal_conv1d_fwd_kernel", "causal_conv1d_bwd_kernel", "l2norm_fwd_kernel", "l2norm_bwd_kernel")
NB_POSITION = {"causal_conv1d_fwd_kernel": 2, "causal_conv1d_bwd_kernel": 2, "l2norm_fwd_kernel": 1, "l2norm_bwd_kernel": 1}
ENTRY = re.compile(
    r"Triton autotuning for function (\w+),\s*with key as \((.*?)\),\s*finished after ([0-9.]+)s,\s*best config selected: (.*?);",
    re.S,
)
META = {"num_warps", "num_ctas", "num_stages", "maxnreg"}


def parse_key(text):
    # Keys are ints, bools and dtype strings, printed as a Python tuple.
    values = []
    for item in (part.strip() for part in text.split(",") if part.strip()):
        if item in ("True", "False"):
            values.append(item == "True")
        elif re.fullmatch(r"-?\d+", item):
            values.append(int(item))
        else:
            values.append(item.strip("'\""))
    return values


def parse_config(text):
    kwargs, meta = {}, {}
    for item in (part.strip() for part in text.split(",") if part.strip()):
        name, value = (s.strip() for s in item.split(":", 1))
        parsed = None if value == "None" else int(value)
        (meta if name in META else kwargs)[name] = parsed
    return {"kwargs": kwargs, "num_warps": meta["num_warps"], "num_stages": meta["num_stages"],
            "num_ctas": meta.get("num_ctas", 1), "maxnreg": meta.get("maxnreg")}


def key_hash(key):
    # Same serialization as fla.ops.utils.cache.AutotuneKey.key_hash.
    return hashlib.md5(json.dumps(key, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS))
    args = parser.parse_args()

    observed = {name: [] for name in args.kernels}
    for log in args.logs:
        for name, key, seconds, config in ENTRY.findall(log.read_text(errors="replace")):
            if name in observed:
                observed[name].append({"autotune_key": parse_key(key), "config": parse_config(config),
                                       "tuning_seconds": float(seconds), "log": log.name})
    args.out.mkdir(parents=True, exist_ok=False)
    summary = {}
    for name, entries in observed.items():
        if not entries:
            raise SystemExit(f"no autotune result for {name} in the logs")
        nb = NB_POSITION.get(name)
        largest = max(entries, key=lambda e: e["autotune_key"][nb] if nb is not None else 0)
        document = {
            "kernel_name": name,
            "triton_version": None,
            "default_config": largest["config"],
            "autotune_entries": {key_hash(e["autotune_key"]): {"autotune_key": e["autotune_key"], "config": e["config"]}
                                 for e in entries},
            "provenance": {"source_logs": sorted({e["log"] for e in entries}),
                           "default_from_key": largest["autotune_key"],
                           "observed": [[e["autotune_key"], e["config"], e["tuning_seconds"]] for e in entries]},
        }
        (args.out / f"{name}.json").write_text(json.dumps(document, indent=1) + "\n")
        summary[name] = {"default_config": largest["config"], "keys": len(entries),
                         "tuning_seconds_observed": round(sum(e["tuning_seconds"] for e in entries), 1)}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
