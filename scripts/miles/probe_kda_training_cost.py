"""Bounded forward/backward feasibility and timing for KDA-only FP32."""

import argparse
import json
from pathlib import Path

import torch
from fla.ops import kda
from scripts.miles import selective_precision_runtime as runtime

from open_instruct.miles import fla_compat


def run(args):
    fla_compat.install_kda_triton_compat()
    runtime.strict_arithmetic()
    capture = torch.load(args.inputs, weights_only=False)[0]["inputs"]
    report = {"scope": "Single captured KDA layer, forward plus backward; not full training throughput", "cases": {}}
    for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        inputs = {}
        for key in ("q", "k", "v", "g", "beta", "A_log", "dt_bias"):
            kind = dtype if key in ("q", "k", "v", "g") else torch.float32
            inputs[key] = capture[key].cuda().to(kind).detach().requires_grad_(True)
        timings = []
        torch.manual_seed(11)
        grad = torch.randn_like(inputs["v"], dtype=torch.bfloat16)
        for repeat in range(7):
            for value in inputs.values():
                value.grad = None
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            output, _ = kda.chunk_kda(**inputs, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True)
            output.bfloat16().backward(grad)
            end.record()
            end.synchronize()
            if repeat >= 2:
                timings.append(start.elapsed_time(end))
        gradients = {}
        for key, value in inputs.items():
            if value.grad is None or not torch.isfinite(value.grad).all():
                raise ValueError(f"Missing or nonfinite {name} gradient: {key}")
            gradients[key] = {"dtype": str(value.grad.dtype), "norm": value.grad.float().norm().item()}
        report["cases"][name] = {"milliseconds": timings, "gradients": gradients}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("KDA_BACKWARD", name, json.dumps(report["cases"][name]), flush=True)
        del output, inputs, grad
    for dtype in (torch.bfloat16,):
        x = torch.randn(4, 128, device="cuda", dtype=dtype, requires_grad=True)
        weight = torch.randn(16, 128, device="cuda", dtype=dtype, requires_grad=True)
        try:
            torch.mm(x, weight.T, out_dtype=torch.float32).sum().backward()
            report["fp32_output_gemm_backward"] = {"supported": True, "finite": bool(torch.isfinite(x.grad).all())}
        except RuntimeError as error:
            report["fp32_output_gemm_backward"] = {"supported": False, "error": str(error)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
