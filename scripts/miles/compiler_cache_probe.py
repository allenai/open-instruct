"""One tiny real Triton kernel; records compiler writes in a fresh process."""

import argparse
import json
import time
from pathlib import Path
from unittest import mock

import torch
import triton
from triton import language as tl
from triton.runtime import cache as triton_cache


@triton.jit
def add_kernel(left, right, output, count: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(left + offsets, offsets < count) + tl.load(right + offsets, offsets < count)
    tl.store(output + offsets, values, offsets < count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-compiler-writes", choices=("present", "absent"), required=True)
    args = parser.parse_args()
    writes = []
    original = triton_cache.FileCacheManager.put

    def record_put(self, data, filename, binary=True):
        writes.append(filename)
        return original(self, data, filename, binary=binary)

    left = torch.arange(8192, device="cuda", dtype=torch.float32)
    right = torch.ones_like(left)
    output = torch.empty_like(left)
    torch.cuda.synchronize()
    started = time.monotonic()
    with mock.patch.object(triton_cache.FileCacheManager, "put", record_put):
        add_kernel[(32,)](left, right, output, left.numel(), BLOCK=256)
        torch.cuda.synchronize()
    seconds = time.monotonic() - started
    torch.testing.assert_close(output, left + right, rtol=0, atol=0)
    report = {
        "schema_version": 1,
        "numerically_exact": True,
        "first_launch_seconds": seconds,
        "compiler_writes": writes,
        "expected_compiler_writes": args.expect_compiler_writes,
        "triton_version": triton.__version__,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    report["valid"] = bool(writes) == (args.expect_compiler_writes == "present")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    if not report["valid"]:
        raise RuntimeError("Actual Triton compiler writes did not match requested cold/restored mode")


if __name__ == "__main__":
    main()
