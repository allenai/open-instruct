"""Standalone Olmo SGLang server with benchmark-only timing hooks."""

import argparse
import importlib
import json
import os
from pathlib import Path

from scripts.miles import topk_instrumentation

# Multiprocessing spawn executes this module again, installing hooks in workers.
if os.environ.get("TOPK_BENCH_CONTROL"):
    topk_instrumentation.install()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    importlib.import_module("olmo_sglang").register()
    server_args = importlib.import_module("sglang.srt.server_args").ServerArgs(
        model_path=args.model,
        host="127.0.0.1",
        port=30000,
        trust_remote_code=True,
        tp_size=1,
        dtype="bfloat16",
        context_length=1024,
        mem_fraction_static=0.75,
        max_running_requests=32,
        max_total_tokens=32768,
        max_mamba_cache_size=64,
        disable_radix_cache=True,
        attention_backend="fa3",
        sampling_backend="pytorch",
        cuda_graph_backend_decode="full",
        cuda_graph_backend_prefill="disabled",
        cuda_graph_max_bs_decode=32,
        enable_metrics=True,
        random_seed=17,
    )
    server_args.resolve_once()
    (args.output / "server-args.json").write_text(json.dumps(server_args.resolved_dict(), indent=2, default=str))
    importlib.import_module("sglang.launch_server").run_server(server_args)


if __name__ == "__main__":
    main()
