"""Matched local export/packing comparison; excludes transport and serving."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from olmo_core.config import DType
from olmo_core.nn import attention
from olmo_core.nn.hf import config as hf_config_utils
from olmo_core.nn.moe.v2 import olmo3
from transformers import AutoModelForCausalLM


def measure(model, hf, stream, buffer_bytes):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial = torch.cuda.memory_allocated()
    started = time.perf_counter()
    iterator = (
        olmo3.iter_olmo3_moe_hf_state(model, hf)
        if stream
        else olmo3.gather_olmo3_moe_hf_state(model, hf, cpu=True).items()
    )
    bucket, size, count = [], 0, 0
    for _, tensor in iterator:
        if bucket and size + tensor.nbytes > buffer_bytes:
            torch.cuda.synchronize()
            bucket, size = [], 0
        bucket.append(tensor.to("cuda").contiguous())
        size += tensor.nbytes
        count += 1
    torch.cuda.synchronize()
    return dict(
        seconds=time.perf_counter() - started,
        extra_peak_gpu_bytes=torch.cuda.max_memory_allocated() - initial,
        tensors=count,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--buffer-mib", type=int, default=32)
    args = parser.parse_args()
    hf_config_utils._register_olmo3moe_auto_classes()
    reference = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    hf = reference.config
    config = olmo3.build_olmo3_moe_config_from_hf_config(
        hf, dtype=DType.bfloat16, attention_backend=attention.AttentionBackendName.torch
    )
    model = config.build(init_device="cpu")
    olmo3.load_olmo3_moe_hf_state(model, hf, reference.state_dict())
    model.cuda()
    for name, tensor in olmo3.iter_olmo3_moe_hf_state(model, hf):
        torch.testing.assert_close(tensor.cpu(), reference.state_dict()[name], rtol=0, atol=0, check_dtype=False)
    del reference, tensor
    samples = {"cpu_staging": [], "streaming": []}
    for iteration in range(args.repetitions + 2):
        # Alternate the order to avoid consistently favoring a warmed second path.
        modes = [False, True] if iteration % 2 == 0 else [True, False]
        for stream in modes:
            result = measure(model, hf, stream, args.buffer_mib * 1024 * 1024)
            if iteration >= 2:
                samples["streaming" if stream else "cpu_staging"].append(result)
    report = dict(
        checkpoint=str(args.checkpoint),
        device=torch.cuda.get_device_name(),
        buffer_mib=args.buffer_mib,
        parameters=sum(p.numel() for p in model.parameters()),
        samples=samples,
        medians={
            mode: {key: statistics.median(row[key] for row in rows) for key in rows[0]}
            for mode, rows in samples.items()
        },
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["medians"], indent=2))


if __name__ == "__main__":
    main()
