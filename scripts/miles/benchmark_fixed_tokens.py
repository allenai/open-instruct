"""Measure warmed, fixed-length SGLang generation separately from answer accuracy.

Use a qualification options.json from the same image/model/precision. Outputs
are synthetic performance probes and must never be included in accuracy scores.
"""

import argparse
import hashlib
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
    parser.add_argument("--context-length", type=int, default=18434)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    options = json.loads(args.options.read_text())
    options["context_length"] = args.context_length
    if args.context_length < 2048 + 16384 + 2:
        parser.error("Context must include two scheduler bookkeeping tokens")
    provenance = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "qualification_options_sha256": hashlib.sha256(args.options.read_bytes()).hexdigest(),
        "job_id": os.environ.get("BEAKER_JOB_ID"),
        "timing": "monotonic streaming; load/compile/warmup/cache flush excluded",
        "context_length": args.context_length,
    }
    measurements = args.output / "measurements.jsonl"
    records = [json.loads(line) for line in measurements.read_text().splitlines()] if measurements.exists() else []
    keys = ("prompt_tokens", "output_tokens_per_request", "concurrency", "repeat")
    expected = {
        (p, o, c, r) for p in (512, 2048) for o in (256, 4096, 16384) for c in (1, 8, 16) for r in range(args.repeats)
    }
    completed = {tuple(record[key] for key in keys) for record in records}
    if len(completed) != len(records) or not completed <= expected:
        raise ValueError("Invalid or duplicate saved benchmark cases")
    for record in records:
        if (
            record["precision"] != args.precision
            or record["context_length"] != args.context_length
            or record["provenance"]["qualification_options_sha256"] != provenance["qualification_options_sha256"]
            or len(record["completion_counts"]) != record["concurrency"]
            or set(record["completion_counts"].values()) != {record["output_tokens_per_request"]}
        ):
            raise ValueError("Saved benchmark provenance/counts differ")
    if completed == expected and (args.output / "complete.json").exists():
        return
    (args.output / f"provenance-{os.environ.get('BEAKER_JOB_ID', 'local')}.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
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
        with measurements.open("a") as stream:
            for prompt_length in (512, 2048):
                tokens = (sentence * (prompt_length // len(sentence) + 1))[:prompt_length]
                for output_length in (256, 4096, 16384):
                    for concurrency in (1, 8, 16):
                        for repeat in range(args.repeats):
                            if (prompt_length, output_length, concurrency, repeat) in completed:
                                continue
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
                                "context_length": args.context_length,
                                "completion_counts": counts,
                                "provenance": provenance,
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
                            os.fsync(stream.fileno())
        (args.output / "complete.json").write_text(
            json.dumps({"cases": 18 * args.repeats, "options": options}, indent=2) + "\n"
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
