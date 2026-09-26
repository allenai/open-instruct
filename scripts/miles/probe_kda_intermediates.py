"""Round one FP32 KDA intermediate at a time on identical captured inputs.

This is a forward-only attribution probe, not an alternative production kernel.
It measures each rounding intervention against the all-FP32 chunk calculation.
"""

import argparse
import importlib
import json
from pathlib import Path

import torch
from fla.ops import kda
from scripts.miles import probe_kda_boundaries as boundaries
from scripts.miles import selective_precision_runtime as runtime

from open_instruct.miles.training import fla_compat

STAGES = ("normalization", "w", "u", "kg", "Aqk", "h", "v_new", "intra_all", "state_all")


def run(args):
    fla_compat.install_kda_triton_compat()
    runtime.strict_arithmetic()
    captures = torch.load(args.inputs, weights_only=False)
    chunk = importlib.import_module("fla.ops.kda.chunk")
    forward = importlib.import_module("fla.ops.kda.chunk_fwd")
    original_norm = chunk.l2norm_fwd
    original_intra = forward.chunk_kda_fwd_intra
    original_state = forward.chunk_gated_delta_rule_fwd_h
    control = {"stage": None}

    def rounded(value):
        return value.bfloat16().float() if value is not None else None

    def norm(*args, **kwargs):
        output, rstd = original_norm(*args, **kwargs)
        return (rounded(output) if control["stage"] == "normalization" else output), rstd

    def intra(*args, **kwargs):
        values = list(original_intra(*args, **kwargs))
        for i, name in enumerate(("w", "u", "qg", "kg", "Aqk", "Akk")):
            if control["stage"] in (name, "intra_all"):
                values[i] = rounded(values[i])
        return tuple(values)

    def state(*args, **kwargs):
        values = list(original_state(*args, **kwargs))
        for i, name in enumerate(("h", "v_new")):
            if control["stage"] in (name, "state_all"):
                values[i] = rounded(values[i])
        return tuple(values)

    chunk.l2norm_fwd = norm
    forward.chunk_kda_fwd_intra = intra
    forward.chunk_gated_delta_rule_fwd_h = state
    report = {"scope": __doc__, "cases": []}
    with torch.no_grad():
        for capture in captures:
            inputs = {n: capture["inputs"][n].cuda().float() for n in ("q", "k", "v", "g", "beta", "A_log", "dt_bias")}
            control["stage"] = None
            expected, _ = kda.chunk_kda(**inputs, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True)
            case = {"row": capture["row"], "layer": capture["layer"], "stages": {}}
            for stage in STAGES:
                control["stage"] = stage
                actual, _ = kda.chunk_kda(**inputs, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True)
                start = capture["prefix"]
                stats = boundaries.differences(actual[:, start:], expected[:, start:])
                stats["bf16_output"] = boundaries.differences(
                    actual[:, start:].bfloat16(), expected[:, start:].bfloat16()
                )
                case["stages"][stage] = stats
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print("KDA_INTERMEDIATES", json.dumps(case), flush=True)
    chunk.l2norm_fwd = original_norm
    forward.chunk_kda_fwd_intra = original_intra
    forward.chunk_gated_delta_rule_fwd_h = original_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
