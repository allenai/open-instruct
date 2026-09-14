"""Fixed-weight TP1 batch sweep, independent of the trainer's completed queue."""

import argparse
import gc
import json
import time
from pathlib import Path

import sglang
import torch
from olmo_sglang import register
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--output-tokens", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if any(c <= 0 for c in args.concurrencies) or args.output_tokens <= 0 or args.repeats < 2:
        parser.error("Use positive concurrency/output lengths and at least two repeats")
    args.output.mkdir(parents=True, exist_ok=True)
    register()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    largest = max(args.concurrencies)
    engine = None
    report = {
        "scope": "Frozen policy, fixed-length decode batches; no trainer, HTTP fleet router, refresh or reward. Includes routed-expert/logprob returns. Not a GSM8K learning test.",
        "model": args.model,
        "concurrencies": args.concurrencies,
        "output_tokens": args.output_tokens,
        "repeats": args.repeats,
        "gpu": torch.cuda.get_device_name(),
        "measurements": [],
        "status": "starting",
    }

    def save():
        (args.output / "inference-capacity.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        started = time.perf_counter()
        engine = sglang.Engine(
            model_path=args.model,
            trust_remote_code=True,
            dtype="bfloat16",
            tp_size=1,
            skip_tokenizer_init=True,
            context_length=6144,
            disable_radix_cache=False,
            mamba_radix_cache_strategy="extra_buffer",
            attention_backend="triton",
            sampling_backend="pytorch",
            max_running_requests=largest,
            max_total_tokens=largest * 6144,
            max_mamba_cache_size=largest * 8,
            cuda_graph_backend_decode="full",
            cuda_graph_max_bs_decode=largest,
            cuda_graph_backend_prefill="disabled",
            chunked_prefill_size=8192,
            mem_fraction_static=0.9,
            enable_return_routed_experts=True,
            random_seed=17,
        )
        report["startup_seconds"] = time.perf_counter() - started
        report["status"] = "measuring"
        save()
        for concurrency in args.concurrencies:
            # Distinct 512-token inputs avoid an artificial single shared-prefix batch.
            prompts = []
            for index in range(concurrency):
                tokens = tokenizer.encode(
                    f"Problem {index}: "
                    + "A shop buys and sells apples. Find the total cost and explain each step. " * 100,
                    add_special_tokens=False,
                )
                prompts.append(tokens[:512])
            for repeat in range(args.repeats + 1):
                # First short run warms batch-specific kernels; repeat zero still
                # records any remaining long-context cold cost and is not steady evidence.
                output_tokens = 256 if repeat == 0 else args.output_tokens
                engine.flush_cache()
                started = time.perf_counter()
                outputs = engine.generate(
                    input_ids=prompts,
                    sampling_params={"temperature": 1.0, "max_new_tokens": output_tokens, "ignore_eos": True},
                    return_logprob=True,
                    logprob_start_len=-1,
                    return_routed_experts=True,
                )
                elapsed = time.perf_counter() - started
                lengths = [item["meta_info"]["completion_tokens"] for item in outputs]
                if len(outputs) != concurrency or lengths != [output_tokens] * concurrency:
                    raise RuntimeError("Inference sweep returned incomplete or shortened batches")
                record = dict(
                    concurrency=concurrency,
                    repeat=repeat,
                    warmup=repeat <= 1,
                    started_unix=time.time() - elapsed,
                    seconds=elapsed,
                    input_tokens=sum(map(len, prompts)),
                    output_tokens=sum(lengths),
                    output_tokens_per_second=sum(lengths) / elapsed,
                    scope="One concurrent fixed-length batch, including prefill and result delivery",
                )
                report["measurements"].append(record)
                save()
                print(json.dumps(record), flush=True)
                del outputs
                gc.collect()
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
