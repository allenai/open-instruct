"""Measure optional Core-compatible serving on frozen real RL prompts.

Separate serving subprocesses isolate mode selection; Core teacher-forces every
retained rollout. This benchmarks generation and probabilities, not learning.
"""

import argparse
import collections
import hashlib
import json
import os
import time
from pathlib import Path

import sglang
import torch
from olmo_core import config as core_config
from olmo_core.nn import attention
from olmo_core.nn.moe.v2 import olmo3
from olmo_sglang import register
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

from open_instruct.miles import fla_compat

DOMAINS = ("math", "code", "ifeval", "general")
SOURCE_SHA256 = "fde6da774f735ea8d3720598f85ecbd613d5fcf0533f85dd5d8997fedbe93805"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def select_rows(rows, per_domain):
    buckets = collections.defaultdict(list)
    seen = set()
    for row in rows:
        domain = row["metadata"].get("domain")
        if domain not in DOMAINS:
            verifier = row["metadata"].get("verifiers", [{}])[0].get("name", "")
            domain = {"general-quality": "general", "general-quality_ref": "general", "code_stdio": "code"}.get(
                verifier, verifier
            )
        if domain in DOMAINS and len(buckets[domain]) < per_domain and row["input"] not in seen:
            buckets[domain].append(row)
            seen.add(row["input"])
    if any(len(buckets[name]) != per_domain for name in DOMAINS):
        raise ValueError(f"Insufficient domain coverage: { {k: len(v) for k, v in buckets.items()} }")
    return [(name, buckets[name][index]) for index in range(per_domain) for name in DOMAINS]


def prepare(args):
    raw = args.data.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Frozen workload source digest changed")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    selected = []
    for domain, row in select_rows(rows, args.per_domain):
        ids = tokenizer(row["input"], add_special_tokens=False)["input_ids"]
        if not ids or len(ids) > 2048:
            raise ValueError(f"Workload prompt exceeds 2048-token bound: {domain}, {len(ids)}")
        selected.append(
            {
                "domain": domain,
                "identity": row["metadata"].get("prepared_sample_id"),
                "input": row["input"],
                "input_ids": ids,
                "metadata": row["metadata"],
            }
        )
    write_json(
        args.output,
        {"source": str(args.data), "source_sha256": digest, "tokenizer_source": args.model, "rows": selected},
    )
    print("WORKLOAD_PREPARED", len(selected), [len(r["input_ids"]) for r in selected], flush=True)


def serve(args):
    os.environ["OLMO_SGLANG_CORE_COMPAT"] = "1" if args.mode == "core" else "0"
    register()
    rows = json.loads(args.samples.read_text())["rows"]
    engine_args = dict(
        model_path=args.model,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tp_size=1,
        attention_backend="triton",
        sampling_backend="pytorch",
        random_seed=20260923,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        context_length=4096,
        max_total_tokens=16384,
        max_running_requests=args.batch_size,
        max_mamba_cache_size=16,
        chunked_prefill_size=8192,
        max_prefill_tokens=8192,
        mem_fraction_static=0.65,
        skip_server_warmup=True,
    )
    started = time.perf_counter()
    engine = sglang.Engine(**engine_args)
    load_seconds = time.perf_counter() - started
    report = {
        "model": args.model,
        "mode": args.mode,
        "engine_args": engine_args,
        "gpu": torch.cuda.get_device_name(),
        "samples_sha256": hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        "load_seconds": load_seconds,
        "warmup": [],
        "batches": [],
        "rollouts": [],
        "forced": [],
    }
    try:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            engine.generate(
                input_ids=[r["input_ids"] for r in batch],
                sampling_params={
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
            )
            report["warmup"].append([start, len(batch)])
        for repeat in range(args.repeats):
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                started = time.perf_counter()
                results = engine.generate(
                    input_ids=[r["input_ids"] for r in batch],
                    sampling_params={
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "top_k": -1,
                        "max_new_tokens": args.tokens,
                        "ignore_eos": True,
                    },
                    return_logprob=True,
                )
                elapsed = time.perf_counter() - started
                token_count = sum(len(result["output_ids"]) for result in results)
                report["batches"].append(
                    {
                        "repeat": repeat,
                        "start": start,
                        "seconds": elapsed,
                        "generated_tokens": token_count,
                        "tokens_per_second": token_count / elapsed,
                    }
                )
                for offset, result in enumerate(results):
                    lp = result["meta_info"]["output_token_logprobs"]
                    if [v[1] for v in lp] != result["output_ids"]:
                        raise ValueError("Serving probability/token alignment changed")
                    report["rollouts"].append(
                        {
                            "row": start + offset,
                            "repeat": repeat,
                            "output_ids": result["output_ids"],
                            "logprobs": [v[0] for v in lp],
                            "meta_info": result["meta_info"],
                        }
                    )
                write_json(args.output, report)
                print("BENCH_BATCH", args.mode, repeat, start, token_count, elapsed, flush=True)
        if args.reference_rollouts:
            baseline = json.loads(args.reference_rollouts.read_text())
            # One matched trajectory per domain; distinguish prefill scoring from cached decode.
            for rollout in baseline["rollouts"][:4]:
                row = rollout["row"]
                ids = rows[row]["input_ids"] + rollout["output_ids"]
                result = engine.generate(
                    input_ids=ids,
                    sampling_params={"temperature": 0, "max_new_tokens": 1},
                    return_logprob=True,
                    logprob_start_len=0,
                )
                values = result["meta_info"]["input_token_logprobs"]
                if len(values) != len(ids) or [v[1] for v in values] != ids:
                    raise ValueError("Forced-prefix input probability alignment changed")
                report["forced"].append(
                    {
                        "row": row,
                        "output_ids": rollout["output_ids"],
                        "logprobs": [v[0] for v in values[len(rows[row]["input_ids"]) :]],
                    }
                )
        write_json(args.output, report)
    finally:
        engine.shutdown()


def summarize_delta(actual, reference):
    delta = actual.double() - reference.double()
    if not torch.isfinite(delta).all():
        raise ValueError("Nonfinite probability comparison")
    absolute = delta.abs()
    ratios = (-delta).exp()
    return {
        "count": delta.numel(),
        "max_abs": absolute.max().item(),
        "mean_abs": absolute.mean().item(),
        "p95_abs": absolute.quantile(0.95).item(),
        "p99_abs": absolute.quantile(0.99).item(),
        "mean_serving_minus_core": delta.mean().item(),
        "ratio_min": ratios.min().item(),
        "ratio_max": ratios.max().item(),
        "ratio_p01": ratios.quantile(0.01).item(),
        "ratio_p99": ratios.quantile(0.99).item(),
        "ratio_outside_10pct_fraction": ((ratios < 0.9) | (ratios > 1.1)).double().mean().item(),
        "ratio_outside_20pct_fraction": ((ratios < 0.8) | (ratios > 1.2)).double().mean().item(),
    }


def score(args):
    fla_compat.install_kda_triton_compat()
    samples = json.loads(args.samples.read_text())
    rows = samples["rows"]
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    core = olmo3.build_olmo3_moe_config_from_hf_config(
        config,
        dtype=core_config.DType.bfloat16,
        attention_backend=attention.AttentionBackendName.torch,
        router_aux_loss_weight=0.0,
        router_z_loss_weight=0.0,
    )
    core.recompute_each_block = False
    model = core.build(init_device="meta")
    model.init_weights(max_seq_len=4096, max_local_microbatch_size=4096, device=torch.device("cuda"))
    state = {}
    for shard in sorted(Path(args.model).glob("*.safetensors")):
        part = load_file(shard)
        if state.keys() & part.keys():
            raise ValueError("Duplicate checkpoint tensor")
        state.update(part)
    olmo3.load_olmo3_moe_hf_state(model, config, state)
    del state
    model.eval()
    report = {"model": args.model, "gpu": torch.cuda.get_device_name(), "comparisons": {}}
    for path in args.serving_reports:
        serving = json.loads(path.read_text())
        if serving["samples_sha256"] != hashlib.sha256(args.samples.read_bytes()).hexdigest():
            raise ValueError("Serving and scoring prompt identities differ")
        combined, reference, by_domain = [], [], collections.defaultdict(lambda: ([], []))
        output = {"rows": [], "forced": [], "domain": {}}
        with torch.no_grad():
            for kind in ["rollouts", "forced"]:
                for rollout in serving[kind]:
                    row = rows[rollout["row"]]
                    ids = row["input_ids"] + rollout["output_ids"]
                    logits = model(torch.tensor([ids], device="cuda"))[0]
                    chosen = []
                    start = len(row["input_ids"]) - 1
                    for offset in range(start, len(ids) - 1, 128):
                        chunk = logits[offset : min(offset + 128, len(ids) - 1)].float()
                        tokens = torch.tensor(ids[offset + 1 : offset + 1 + len(chunk)], device="cuda")
                        lp = chunk.gather(-1, tokens[:, None]).squeeze(-1) - chunk.logsumexp(-1)
                        chosen.append(lp.cpu())
                    actual, expected = torch.tensor(rollout["logprobs"]), torch.cat(chosen)
                    del logits
                    if len(actual) != len(expected):
                        raise ValueError("Core and serving token counts differ")
                    detail = {
                        "row": rollout["row"],
                        "repeat": rollout.get("repeat"),
                        "domain": row["domain"],
                        "statistics": summarize_delta(actual, expected),
                        "core_logprobs": expected.tolist(),
                    }
                    output["rows" if kind == "rollouts" else "forced"].append(detail)
                    if kind == "rollouts":
                        combined.append(actual)
                        reference.append(expected)
                        by_domain[row["domain"]][0].append(actual)
                        by_domain[row["domain"]][1].append(expected)
        output["aggregate"] = summarize_delta(torch.cat(combined), torch.cat(reference))
        for name, (a, b) in by_domain.items():
            output["domain"][name] = summarize_delta(torch.cat(a), torch.cat(b))
        duration = sum(b["seconds"] for b in serving["batches"])
        count = sum(b["generated_tokens"] for b in serving["batches"])
        output["throughput"] = {
            "generated_tokens": count,
            "seconds": duration,
            "tokens_per_second": count / duration,
            "batches": serving["batches"],
        }
        report["comparisons"][serving["mode"]] = output
        write_json(args.output, report)
        print(
            "CORE_BENCH_RESULT",
            serving["mode"],
            json.dumps(output["aggregate"]),
            output["throughput"]["tokens_per_second"],
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "serve", "score"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--per-domain", type=int, default=4)
    parser.add_argument("--mode", choices=["default", "core"], default="default")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reference-rollouts", type=Path)
    parser.add_argument("--serving-reports", type=Path, nargs="+")
    args = parser.parse_args()
    {"prepare": prepare, "serve": serve, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
