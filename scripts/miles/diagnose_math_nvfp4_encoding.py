"""Replay exact pinned expert quantization under load/publication dtype contexts.

Diagnostic only: this cannot qualify the real Core publication transport.
"""

import argparse
import importlib
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open
from scripts.miles import qualify_math_weight_publication as gate

# Exact values from DefaultModelLoader.load_weights_only in the pinned image.
# The original loader restores them before serving, so a direct publication
# through model.load_weights otherwise sees a different quantization recipe.
LOADER_ENV = {
    "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1",
    "FLASHINFER_NVFP4_4OVER6": "1",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "0",
    "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
    "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
}


@contextmanager
def encoding_context(mode):
    if mode == "ambient":
        yield
    elif mode == "pinned_loader":
        utils = importlib.import_module("sglang.srt.utils")
        with utils.temp_set_env(**LOADER_ENV):
            yield
    else:
        raise ValueError(f"Unknown encoding context: {mode}")


def replay(args):
    torch.cuda.set_device(args.gpu)
    registration = importlib.import_module("olmo_sglang")
    registration.register()
    method = importlib.import_module("sglang.srt.layers.quantization.nvfp4_online").ModelOptNvFp4OnlineFusedMoEMethod
    rows = []
    original_dtype = torch.get_default_dtype()
    try:
        with safe_open(args.model / "model.safetensors", framework="pt", device=f"cuda:{args.gpu}") as source:
            for layer in (1, 8):
                for expert in (0, 1):
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                        weight = source.get_tensor(name).contiguous()
                        cases = (
                            ("pinned_loader", torch.bfloat16),
                            ("ambient", torch.float32),
                            ("pinned_loader", torch.float32),
                            ("ambient", torch.bfloat16),
                        )
                        for index, (mode, dtype) in enumerate(cases):
                            torch.set_default_dtype(dtype)
                            with encoding_context(mode):
                                encoded = method._quantize_weight_nvfp4(weight)
                            row = {
                                "name": name,
                                "iteration": index,
                                "context": mode,
                                "loader_environment": LOADER_ENV if mode == "pinned_loader" else None,
                                "default_dtype": str(dtype),
                                "input": gate.tensor_fingerprint(weight),
                                "stride": list(weight.stride()),
                                "provided_scale": None,
                                "encoded": [gate.tensor_fingerprint(value) for value in encoded],
                            }
                            rows.append(row)
                            if layer == 1 and expert == 0 and projection == "gate_proj":
                                torch.save(
                                    {"input": weight.cpu(), "encoded": [value.cpu() for value in encoded]},
                                    args.output / f"gpu{args.gpu}-sample-{index}.pt",
                                )
                            gate.save(args.output / f"gpu{args.gpu}-progress.json", rows)
    finally:
        torch.set_default_dtype(original_dtype)
    gate.save(args.output / f"gpu{args.gpu}-complete.json", {"diagnostic_only": True, "rows": len(rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.gpu is not None:
        replay(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    if torch.cuda.device_count() != 4 or any("B300" not in torch.cuda.get_device_name(i) for i in range(4)):
        raise ValueError("Requires exactly four approved B300 GPUs")
    processes = []
    for gpu in (1, 2, 3):
        with (args.output / f"gpu{gpu}.log").open("w") as log:
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "scripts.miles.diagnose_math_nvfp4_encoding",
                        "--model",
                        str(args.model),
                        "--output",
                        str(args.output),
                        "--gpu",
                        str(gpu),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
    statuses = [process.wait() for process in processes]
    gate.save(args.output / "diagnostic-complete.json", {"diagnostic_only": True, "exit_codes": statuses})
    if any(statuses):
        raise RuntimeError(f"Encoder diagnostic subprocess failures: {statuses}")


if __name__ == "__main__":
    main()
