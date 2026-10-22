"""Check FlashInfer compatibility with the locked SGLang source at image build time.

This checks the targeted FlashInfer upgrade, not every optional SGLang backend.
The inherited Torch/CUDA stack stays fixed; validate FlashInfer's direct runtime
requirements before importing the Blackwell modules without executing a kernel.
"""

import argparse
import importlib
from importlib import metadata
from pathlib import Path

import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def check(sglang_root):
    project = tomllib.loads((sglang_root / "python/pyproject.toml").read_text())
    requirement = next(
        Requirement(value)
        for value in project["project"]["dependencies"]
        if canonicalize_name(Requirement(value).name) == "flashinfer-python"
    )
    installed = metadata.version("flashinfer-python")
    if installed not in requirement.specifier:
        raise ValueError(f"SGLang requires {requirement}, installed flashinfer-python=={installed}")
    cubin = metadata.version("flashinfer-cubin")
    if cubin != installed:
        raise ValueError(f"FlashInfer Python/cubin versions differ: {installed} / {cubin}")
    for value in metadata.requires("flashinfer-python") or []:
        dependency = Requirement(value)
        if dependency.marker and not any(
            dependency.marker.evaluate({"extra": extra}) for extra in ("", *requirement.extras)
        ):
            continue
        actual = metadata.version(dependency.name)
        if actual not in dependency.specifier:
            raise ValueError(f"FlashInfer requires {dependency}, installed {dependency.name}=={actual}")
    for module, symbols in (
        ("dense_bf16_gemm_direct", ("default_tactic", "prefer_direct_bf16_gemm_sm100", "run_direct_dense")),
        ("dense_bf16_gemm_sm100_splitk", ("SplitKTactic", "run_splitk_dense")),
    ):
        imported = importlib.import_module(f"flashinfer.gemm.kernels.{module}")
        for symbol in symbols:
            if not hasattr(imported, symbol):
                raise ImportError(f"{imported.__name__} is missing {symbol}")
    print(f"FlashInfer Python/cubin {installed}: SGLang requirement and Blackwell kernel imports passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sglang_root", type=Path)
    check(parser.parse_args().sglang_root)
