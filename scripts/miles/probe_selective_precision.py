"""Matched cached-token scoring and separate warmed throughput for selective precision."""

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
from scripts.miles import benchmark_core_compat as benchmark
from scripts.miles import probe_lm_head_precision as head_probe
from scripts.miles import selective_precision_runtime as runtime
from transformers import AutoConfig

from open_instruct.miles.training import fla_compat


def serve(args):
    os.environ["OLMO_SGLANG_CORE_COMPAT"] = "rounding"
    os.environ["OLMO_SGLANG_ROUNDING_KERNELS"] = "fused"
    register()
    rows = json.loads(args.samples.read_text())["rows"]
    frozen = json.loads(args.rollouts.read_text())["rollouts"][: args.rows]
    report = {
        "model": args.model,
        "samples_sha256": hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        "kda_variant": os.environ.get("OI_KDA_ABLATION", "baseline"),
        "linear_variant": os.environ.get("OI_LINEAR_ABLATION", "none"),
        "prefill_core": os.environ.get("OI_PREFILL_CORE") == "1",
        "chunk_fp32": os.environ.get("OI_CHUNK_FP32") == "1",
        "fp32_lm_head": args.fp32_lm_head,
        "batches": [],
        "rollouts": [],
        "forced": [],
    }
    engine = sglang.Engine(
        model_path=args.model,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enable_fp32_lm_head=args.fp32_lm_head,
        tp_size=1,
        attention_backend="triton",
        sampling_backend="pytorch",
        random_seed=20260923,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        cuda_graph_backend_decode="full",
        cuda_graph_bs_decode=[1, 2, 4],
        cuda_graph_backend_prefill="disabled",
        context_length=4096,
        max_total_tokens=16384,
        max_running_requests=4,
        max_mamba_cache_size=16,
        chunked_prefill_size=8192,
        max_prefill_tokens=8192,
        mem_fraction_static=0.65,
        skip_server_warmup=True,
    )
    params = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": args.tokens, "ignore_eos": True}
    try:
        # Throughput has ordinary sampling; forcing and its CPU synchronization are excluded.
        for repeat in range(args.repeats + 1):
            for start in range(0, len(frozen), 4):
                batch = frozen[start : start + 4]
                prompts = [rows[r["row"]]["input_ids"] for r in batch]
                began = time.perf_counter()
                results = engine.generate(input_ids=prompts, sampling_params=params, return_logprob=True)
                elapsed = time.perf_counter() - began
                if repeat:
                    count = sum(len(r["output_ids"]) for r in results)
                    report["batches"].append(
                        {
                            "repeat": repeat,
                            "start": start,
                            "seconds": elapsed,
                            "generated_tokens": count,
                            "tokens_per_second": count / elapsed,
                        }
                    )
            benchmark.write_json(args.output, report)
            print("TIMING_REPEAT", repeat, report["batches"][-1:] if repeat else "warmup", flush=True)
        for start in range(0, len(frozen), 4):
            batch = frozen[start : start + 4]
            prompts = [rows[r["row"]]["input_ids"] for r in batch]
            settings = [
                {
                    **params,
                    "max_new_tokens": len(r["output_ids"][: args.tokens]),
                    "custom_params": {"prefix_length": len(prompt), "forced_ids": r["output_ids"][: args.tokens]},
                }
                for r, prompt in zip(batch, prompts, strict=True)
            ]
            results = engine.generate(input_ids=prompts, sampling_params=settings, return_logprob=True)
            for original, result in zip(batch, results, strict=True):
                expected_ids = original["output_ids"][: args.tokens]
                if result["output_ids"] != expected_ids:
                    raise ValueError("Cached teacher forcing did not retain exact continuation")
                values = result["meta_info"]["output_token_logprobs"]
                if [v[1] for v in values] != expected_ids:
                    raise ValueError("Forced-token probability alignment changed")
                report["rollouts"].append(
                    {"row": original["row"], "output_ids": expected_ids, "logprobs": [v[0] for v in values]}
                )
            benchmark.write_json(args.output, report)
            print("FIXED_CACHED", start, len(batch), flush=True)
        for rollout in frozen[:4]:
            prompt = rows[rollout["row"]]["input_ids"]
            output_ids = rollout["output_ids"][: args.tokens]
            ids = prompt + output_ids
            result = engine.generate(
                input_ids=ids,
                sampling_params={"temperature": 0, "max_new_tokens": 1},
                return_logprob=True,
                logprob_start_len=0,
            )
            values = result["meta_info"]["input_token_logprobs"]
            if [v[1] for v in values] != ids:
                raise ValueError("Full-prefix probability alignment changed")
            report["forced"].append(
                {"row": rollout["row"], "output_ids": output_ids, "logprobs": [v[0] for v in values[len(prompt) :]]}
            )
        benchmark.write_json(args.output, report)
    finally:
        engine.shutdown()


def score(args):
    torch.set_num_threads(args.cpu_threads)
    print("SCORER_CPU_THREADS", torch.get_num_threads(), flush=True)
    fla_compat.install_kda_triton_compat()
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.chunk_fp32:
        runtime.install_chunk_fp32()
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
        state.update(load_file(shard))
    olmo3.load_olmo3_moe_hf_state(model, config, state)
    del state
    model.eval()
    print("SCORER_MODEL_READY", flush=True)
    versions = {n: p._version for n, p in model.named_parameters()}
    captured = {}

    def capture(module, inputs):
        captured["hidden"] = inputs[0]

    model.lm_head.w_out.register_forward_pre_hook(capture)
    rows = json.loads(args.samples.read_text())["rows"]
    digest = hashlib.sha256(args.samples.read_bytes()).hexdigest()
    cache = {}
    report = {"model": args.model, "samples_sha256": digest, "comparisons": {}}
    with torch.no_grad():
        for path in args.serving_reports:
            serving = json.loads(path.read_text())
            if serving["model"] != args.model or serving["samples_sha256"] != digest:
                raise ValueError("Model/prompt provenance mismatch")
            linear = serving["linear_variant"]
            runtime.set_linear_variant(model, linear, serving=False)
            detail = {"rows": [], "aggregate": {}}
            combined = collections.defaultdict(lambda: ([], []))
            for kind in ("rollouts", "forced"):
                for rollout in serving[kind]:
                    prompt = rows[rollout["row"]]["input_ids"]
                    ids = prompt + rollout["output_ids"]
                    key = (linear, tuple(ids))
                    if key not in cache:
                        logits = model(torch.tensor([ids], device="cuda"))
                        bf16 = head_probe.selected_scores(logits, ids, len(prompt) - 1)
                        del logits
                        hidden = captured.pop("hidden")
                        logits = head_probe.head_logits(hidden, model.lm_head.w_out.weight)
                        fp32 = head_probe.selected_scores(logits, ids, len(prompt) - 1)
                        del logits, hidden
                        cache[key] = {"bf16": bf16, "fp32": fp32}
                    actual = torch.tensor(rollout["logprobs"])
                    detail["rows"].append(
                        {
                            "kind": kind,
                            "row": rollout["row"],
                            "core_logprobs": {n: v.tolist() for n, v in cache[key].items()},
                        }
                    )
                    for head, expected in cache[key].items():
                        combined[f"{kind}/{head}"][0].append(actual)
                        combined[f"{kind}/{head}"][1].append(expected)
            for name, (actual, expected) in combined.items():
                detail["aggregate"][name] = head_probe.probability_statistics(torch.cat(actual), torch.cat(expected))
            report["comparisons"][path.stem] = detail
            benchmark.write_json(args.output, report)
            print("SELECTIVE_RESULT", path.stem, json.dumps(detail["aggregate"]), flush=True)
    if versions != {n: p._version for n, p in model.named_parameters()}:
        raise ValueError("Frozen parameters changed")
    report["parameter_versions_unchanged"] = True
    benchmark.write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("serve", "score"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fp32-lm-head", action="store_true")
    parser.add_argument("--chunk-fp32", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--serving-reports", type=Path, nargs="+")
    args = parser.parse_args()
    {"serve": serve, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
