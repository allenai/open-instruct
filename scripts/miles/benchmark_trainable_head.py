"""Measure the opt-in trainable head independently of model/rollout variability."""

import argparse
import json
from pathlib import Path

import torch
from olmo_core.config import DType
from olmo_core.nn.lm_head import LMHeadConfig


def run(output):
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "cases": []}
    for tokens in (128, 512, 2048):
        for enabled in (False, True):
            head = LMHeadConfig(fp32_output=enabled, dtype=DType.bfloat16, bias=False).build(
                d_model=1024, vocab_size=100278, init_device="cuda"
            )
            torch.nn.init.normal_(head.w_out.weight, std=0.02)
            x = torch.randn(1, tokens, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            labels = torch.randint(0, 100278, (1, tokens), device="cuda")
            times = []
            torch.cuda.reset_peak_memory_stats()
            for repeat in range(25):
                head.zero_grad(set_to_none=True)
                x.grad = None
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                result = head(x, labels=labels)
                result.loss.backward()
                end.record()
                end.synchronize()
                if repeat >= 5:
                    times.append(start.elapsed_time(end))
            assert torch.isfinite(head.w_out.weight.grad).all() and torch.isfinite(x.grad).all()
            case = {
                "tokens": tokens,
                "fp32_output": enabled,
                "milliseconds": times,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "output_dtype": str(result.logits.dtype),
                "weight_dtype": str(head.w_out.weight.dtype),
                "weight_gradient_dtype": str(head.w_out.weight.grad.dtype),
            }
            report["cases"].append(case)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print("HEAD_BENCHMARK", json.dumps(case), flush=True)
            del result, head, x, labels
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
