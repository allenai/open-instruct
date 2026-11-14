"""Measure warmed, fixed-length SGLang generation separately from answer accuracy.

Use a qualification options.json from the same image/model/precision. Outputs
are synthetic performance probes and must never be included in accuracy scores.
"""

import argparse
import importlib
import json
import os
import time
from pathlib import Path

from scripts.miles.qualify_math_quantization import FP8_EXCLUSIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", required=True, type=Path)
    parser.add_argument("--precision", required=True, choices=("bf16", "bf16-control", "fp8", "nvfp4"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    options = json.loads(args.options.read_text())
    os.environ["OLMO_SGLANG_CORE_COMPAT"] = "auto" if args.precision == "bf16" else "off"
    if args.precision == "fp8":
        os.environ["SGLANG_FP8_IGNORED_LAYERS"] = ",".join(FP8_EXCLUSIONS)
    importlib.import_module("olmo_sglang").register()
    engine = importlib.import_module("sglang").Engine(**options)
    tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(options["model_path"])
    sentence = tokenizer.encode("Solve the following mathematics problem step by step. ", add_special_tokens=False)
    try:
        (args.output / "server-info.json").write_text(
            json.dumps(engine.get_server_info(), default=str, indent=2) + "\n"
        )
        engine.generate(
            input_ids=[sentence] * 16, sampling_params={"temperature": 1.0, "max_new_tokens": 128, "ignore_eos": True}
        )
        with (args.output / "measurements.jsonl").open("x") as stream:
            for prompt_length in (512, 2048):
                tokens = (sentence * (prompt_length // len(sentence) + 1))[:prompt_length]
                for output_length in (256, 4096, 16384):
                    for concurrency in (1, 8, 16):
                        for repeat in range(args.repeats):
                            engine.flush_cache()
                            start = time.monotonic()
                            first, last, counts, intervals = {}, {}, {}, []
                            for chunk in engine.generate(
                                input_ids=[tokens] * concurrency,
                                sampling_params={
                                    "temperature": 1.0,
                                    "top_p": 0.95,
                                    "max_new_tokens": output_length,
                                    "ignore_eos": True,
                                },
                                stream=True,
                            ):
                                now = time.monotonic()
                                for item in chunk if isinstance(chunk, list) else [chunk]:
                                    index = item.get("index", 0)
                                    count = item.get("meta_info", {}).get("completion_tokens", 0)
                                    previous = counts.get(index, 0)
                                    if count > previous:
                                        if index not in first:
                                            first[index] = now - start
                                        elif count == previous + 1:
                                            intervals.append(now - last[index])
                                        last[index] = now
                                        counts[index] = count
                            elapsed = time.monotonic() - start
                            if len(counts) != concurrency or any(count != output_length for count in counts.values()):
                                raise ValueError(f"Incomplete fixed-length generation: {counts}")
                            record = {
                                "precision": args.precision,
                                "prompt_tokens": prompt_length,
                                "output_tokens_per_request": output_length,
                                "concurrency": concurrency,
                                "repeat": repeat,
                                "seconds": elapsed,
                                "output_tokens_per_second": sum(counts.values()) / elapsed,
                                "requests_per_second": concurrency / elapsed,
                                "ttft_seconds": list(first.values()),
                                "single_token_interval_seconds": intervals,
                            }
                            stream.write(json.dumps(record) + "\n")
                            stream.flush()
        (args.output / "complete.json").write_text(
            json.dumps({"cases": 18 * args.repeats, "options": options}, indent=2) + "\n"
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
