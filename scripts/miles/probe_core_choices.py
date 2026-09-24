"""Bounded greedy Core/SGLang comparison; run stages in separate GPU processes.

Retains full-sequence and prefix-at-a-time Core scores separately. Neither Core
path uses a serving cache. Full-reference SGLang is not a promise of exact decode.
"""

import argparse
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
from transformers import AutoConfig

from open_instruct.miles import fla_compat


def read_samples(path):
    return json.loads(path.read_text())["rows"][:4]


def serving_record(result, ids, prompt_length, *, forced):
    meta = result["meta_info"]
    key = "input" if forced else "output"
    values = meta[f"{key}_token_logprobs"]
    tops = meta[f"{key}_top_logprobs"]
    if [v[1] for v in values] != ids or len(tops) != len(ids):
        raise ValueError("Token/logprob/top-logprob alignment changed")
    start = prompt_length if forced else 0
    return {
        "output_ids": ids[start:],
        "logprobs": [v[0] for v in values[start:]],
        "top2": tops[start:],
        # A top-k list does not preserve argmax tie-breaking. For generation,
        # use the actual greedy token; forced scores retain the reported top-1
        # representative and must treat equal top probabilities as ambiguous.
        "argmax_ids": [max(v, key=lambda x: x[0])[1] for v in tops[start:]] if forced else ids,
        "top2_tied": [v[0][0] == v[1][0] for v in tops[start:]],
    }


def serve(args):
    if args.mode == "auto":
        os.environ.pop("OLMO_SGLANG_CORE_COMPAT", None)
    else:
        os.environ["OLMO_SGLANG_CORE_COMPAT"] = "full"
    os.environ.pop("OLMO_SGLANG_ROUNDING_KERNELS", None)
    register()
    rows = read_samples(args.samples)
    engine_args = dict(
        model_path=args.model,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        dtype="auto",
        tp_size=1,
        attention_backend="triton",
        sampling_backend="pytorch",
        random_seed=20260924,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        cuda_graph_backend_decode="full" if args.mode == "auto" else "disabled",
        cuda_graph_bs_decode=[1],
        cuda_graph_backend_prefill="disabled",
        context_length=1024,
        max_total_tokens=4096,
        max_running_requests=1,
        max_mamba_cache_size=4,
        chunked_prefill_size=1024,
        max_prefill_tokens=1024,
        mem_fraction_static=0.65,
        skip_server_warmup=True,
    )
    started = time.perf_counter()
    engine = sglang.Engine(**engine_args)
    report = {
        "model": args.model,
        "mode": args.mode,
        "engine_args": engine_args,
        "load_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(),
        "samples_sha256": hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        "rollouts": [],
        "forced": [],
    }
    params = dict(temperature=0, max_new_tokens=args.tokens, ignore_eos=True)
    try:
        for row in rows:
            engine.generate(input_ids=row["input_ids"], sampling_params=params)
        for i, row in enumerate(rows):
            started = time.perf_counter()
            result = engine.generate(
                input_ids=row["input_ids"], sampling_params=params, return_logprob=True, top_logprobs_num=2
            )
            elapsed = time.perf_counter() - started
            record = serving_record(result, result["output_ids"], len(row["input_ids"]), forced=False)
            if len(record["output_ids"]) != args.tokens:
                raise ValueError("Unexpected greedy generation length")
            # Greedy selection can break exact ties differently; retain both IDs.
            record.update(row=i, domain=row["domain"], seconds=elapsed)
            report["rollouts"].append(record)
            benchmark.write_json(args.output, report)
            print("GREEDY", args.mode, i, elapsed, flush=True)
        reference = json.loads(args.reference.read_text()) if args.reference else report
        for rollout in reference["rollouts"]:
            i = rollout["row"]
            prompt_length = len(rows[i]["input_ids"])
            ids = rows[i]["input_ids"] + rollout["output_ids"]
            result = engine.generate(
                input_ids=ids,
                sampling_params={"temperature": 0, "max_new_tokens": 1},
                return_logprob=True,
                top_logprobs_num=2,
                logprob_start_len=0,
            )
            record = serving_record(result, ids, prompt_length, forced=True)
            record.update(row=i, domain=rows[i]["domain"])
            report["forced"].append(record)
            benchmark.write_json(args.output, report)
    finally:
        engine.shutdown()


def core_record(logits, tokens):
    logits = logits.float()
    tokens = torch.tensor(tokens, device=logits.device)
    top, indices = logits.topk(2, dim=-1)
    chosen = logits.gather(-1, tokens[:, None]).squeeze(-1)
    return {
        "logprobs": (chosen - logits.logsumexp(-1)).tolist(),
        "argmax_ids": logits.argmax(-1).tolist(),
        "top2_ids": indices.tolist(),
        "top2_margin": (top[:, 0] - top[:, 1]).tolist(),
        "chosen_margin": (top[:, 0] - chosen).tolist(),
        "chosen_rank": (1 + (logits > chosen[:, None]).sum(-1)).tolist(),
    }


def comparison(serving, core):
    mismatch = [i for i, (s, c) in enumerate(zip(serving["argmax_ids"], core["argmax_ids"], strict=True)) if s != c]
    return {
        "positions": len(core["argmax_ids"]),
        "argmax_disagreements": mismatch,
        "logprobs": benchmark.summarize_delta(torch.tensor(serving["logprobs"]), torch.tensor(core["logprobs"])),
        "core": core,
    }


def score(args):
    fla_compat.install_kda_triton_compat()
    rows = read_samples(args.samples)
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
    model.init_weights(max_seq_len=1024, max_local_microbatch_size=1024, device=torch.device("cuda"))
    state = {}
    for shard in sorted(Path(args.model).glob("*.safetensors")):
        part = load_file(shard)
        if state.keys() & part.keys():
            raise ValueError("Duplicate checkpoint tensor")
        state.update(part)
    olmo3.load_olmo3_moe_hf_state(model, config, state)
    del state
    model.eval()
    report = {"model": args.model, "config": config.to_dict(), "comparisons": {}}
    # Identical trajectories are reused across modes; do not introduce scorer drift.
    cache = {}
    with torch.no_grad():
        for path in args.serving_reports:
            serving = json.loads(path.read_text())
            if (
                serving["model"] != args.model
                or serving["samples_sha256"] != hashlib.sha256(args.samples.read_bytes()).hexdigest()
            ):
                raise ValueError("Model or prompt provenance mismatch")
            output = {"rollouts": [], "forced": []}
            for kind in output:
                for rollout in serving[kind]:
                    row = rows[rollout["row"]]
                    ids = row["input_ids"] + rollout["output_ids"]
                    key = tuple(ids)
                    start = len(row["input_ids"]) - 1
                    if key not in cache:
                        logits = model(torch.tensor([ids], device="cuda"))[0]
                        full = core_record(logits[start:-1], rollout["output_ids"])
                        del logits
                        prefixes = []
                        for offset, token in enumerate(rollout["output_ids"]):
                            logits = model(torch.tensor([ids[: start + offset + 1]], device="cuda"))[0, -1:]
                            prefixes.append(core_record(logits, [token]))
                            del logits
                        prefix = {k: [v for record in prefixes for v in record[k]] for k in full}
                        cache[key] = (full, prefix)
                    full, prefix = cache[key]
                    detail = {"row": rollout["row"], "domain": row["domain"]}
                    detail["full_sequence"] = comparison(rollout, full)
                    detail["prefix_at_a_time"] = comparison(rollout, prefix)
                    output[kind].append(detail)
                    print(
                        "CHOICES",
                        serving["mode"],
                        kind,
                        rollout["row"],
                        len(detail["prefix_at_a_time"]["argmax_disagreements"]),
                        flush=True,
                    )
            report["comparisons"][serving["mode"]] = output
            benchmark.write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["serve", "score"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["auto", "full"], default="auto")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--serving-reports", type=Path, nargs="+")
    args = parser.parse_args()
    {"serve": serve, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
