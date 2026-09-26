"""Replay real frozen-model KDA inputs to isolate chunk/decode rounding boundaries."""

import argparse
import gc
import hashlib
import inspect
import json
import time
from pathlib import Path

import torch
from fla.modules import l2norm
from fla.ops import kda
from olmo_core import config as core_config
from olmo_core.nn import attention
from olmo_core.nn.attention import kda as core_kda
from olmo_core.nn.moe.v2 import olmo3
from safetensors.torch import load_file
from scripts.miles import benchmark_core_compat as benchmark
from scripts.miles import selective_kda_precision as selective
from transformers import AutoConfig

from open_instruct.miles.training import fla_compat


def differences(actual, expected):
    delta = actual.float() - expected.float()
    return {
        "mean_abs": delta.abs().mean().item(),
        "max_abs": delta.abs().max().item(),
        "relative_rms": (delta.square().mean() / expected.float().square().mean().clamp_min(1e-30)).sqrt().item(),
        "equal_fraction": (actual == expected).float().mean().item(),
    }


def replay(function, inputs, prefix, *, fp32=False):
    q, k, v, g = [inputs[n].cuda() for n in ("q", "k", "v", "g")]
    beta = inputs["beta"].cuda()
    raw_beta = inputs["raw_beta"].cuda()
    a_log, bias = inputs["A_log"].cuda(), inputs["dt_bias"].cuda()
    if fp32:
        q, k, v, g, raw_beta = [x.float() for x in (q, k, v, g, raw_beta)]
    common = dict(
        A_log=a_log, dt_bias=bias, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, output_final_state=True
    )
    full, full_state = kda.chunk_kda(q=q, k=k, v=v, g=g, beta=beta, **common)
    _, initial = kda.chunk_kda(
        q=q[:, :prefix], k=k[:, :prefix], v=v[:, :prefix], g=g[:, :prefix], beta=beta[:, :prefix], **common
    )
    state = initial.transpose(-1, -2).contiguous()
    indices = torch.zeros(1, device="cuda", dtype=torch.int64)
    packed = torch.cat([x.flatten(2) for x in (q, k, v)], dim=-1)[0]
    outputs = []
    for t in range(prefix, q.shape[1]):
        outputs.append(
            function(
                packed[t : t + 1],
                g[0, t : t + 1],
                raw_beta[0, t : t + 1],
                a_log=a_log,
                dt_bias=bias,
                scale=q.shape[-1] ** -0.5,
                state=state,
                state_indices=indices,
                num_value_heads=v.shape[2],
                value_dim=v.shape[3],
                allow_neg_eigval=inputs["allow_neg_eigval"],
            )
        )
    actual = torch.cat(outputs, dim=1)
    expected = full[:, prefix:]
    stats = differences(actual, expected)
    stats["state"] = differences(state.transpose(-1, -2), full_state)
    stats["by_128_tokens"] = [
        differences(actual[:, start : start + 128], expected[:, start : start + 128])
        for start in range(0, actual.shape[1], 128)
    ]
    return stats


def run(args):
    fla_compat.install_kda_triton_compat()
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    for name, module in (("chunk", kda.chunk_kda), ("l2norm", l2norm)):
        (args.output / f"runtime-{name}.py").write_text(inspect.getsource(module))
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
    weights = {}
    for shard in sorted(Path(args.model).glob("*.safetensors")):
        weights.update(load_file(shard))
    olmo3.load_olmo3_moe_hf_state(model, config, weights)
    del weights
    model.eval()
    versions = {n: p._version for n, p in model.named_parameters()}
    samples = json.loads(args.samples.read_text())["rows"]
    rollouts = json.loads(args.rollouts.read_text())["rollouts"][: args.rows]
    captures, context = [], {}
    original = core_kda.dispatch_chunk_kda

    def raw_hook(module, inputs, output):
        context["raw_beta"] = output.detach().cpu()

    for module in model.modules():
        if isinstance(module, core_kda.KimiDeltaAttention):
            module.w_b.register_forward_hook(raw_hook)

    def dispatch(**kwargs):
        result = original(**kwargs)
        layer = context["layer"]
        context["layer"] += 1
        if layer in (0, 6, 13):
            saved = {key: value.detach().cpu() for key, value in kwargs.items() if isinstance(value, torch.Tensor)}
            saved.update(raw_beta=context["raw_beta"], allow_neg_eigval=config.linear_allow_neg_eigval)
            captures.append({"row": context["row"], "layer": layer, "prefix": context["prefix"], "inputs": saved})
        return result

    core_kda.dispatch_chunk_kda = dispatch
    with torch.no_grad():
        for rollout in rollouts:
            row = samples[rollout["row"]]
            context.update(row=rollout["row"], layer=0, prefix=len(row["input_ids"]))
            ids = row["input_ids"] + rollout["output_ids"][: args.tokens]
            model(torch.tensor([ids], device="cuda"))
            print("CAPTURED", context["row"], context["layer"], len(ids), flush=True)
    core_kda.dispatch_chunk_kda = original
    assert versions == {n: p._version for n, p in model.named_parameters()}
    torch.save(captures, args.output / "inputs.pt")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    report = {
        "model": args.model,
        "samples_sha256": hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "parameter_versions_unchanged": True,
        "cases": [],
    }
    functions = {v: selective.load_variant(v, args.output / "kernels") for v in selective.VARIANTS}
    with torch.no_grad():
        for capture in captures:
            case = {k: v for k, v in capture.items() if k != "inputs"}
            case["variants"] = {}
            for variant, (function, digest) in functions.items():
                start = time.perf_counter()
                stats = replay(function, capture["inputs"], capture["prefix"])
                case["variants"][variant] = {**stats, "source_sha256": digest}
                print(
                    "KDA_COMPONENT",
                    case["row"],
                    case["layer"],
                    variant,
                    json.dumps(stats),
                    "seconds",
                    time.perf_counter() - start,
                    flush=True,
                )
            report["cases"].append(case)
            benchmark.write_json(args.output / "report.json", report)


def replay_saved(args):
    fla_compat.install_kda_triton_compat()
    captures = torch.load(args.saved_inputs, weights_only=False)
    args.output.mkdir(parents=True, exist_ok=True)
    function, digest = selective.load_variant("baseline", args.output / "kernels")
    report = {
        "input_sha256": hashlib.sha256(args.saved_inputs.read_bytes()).hexdigest(),
        "kernel_sha256": digest,
        "cases": [],
    }
    with torch.no_grad():
        for capture in captures:
            case = {k: v for k, v in capture.items() if k != "inputs"}
            case["bf16"] = replay(function, capture["inputs"], capture["prefix"])
            case["fp32"] = replay(function, capture["inputs"], capture["prefix"], fp32=True)
            report["cases"].append(case)
            benchmark.write_json(args.output / "report.json", report)
            print("KDA_KERNEL_PRECISION", json.dumps(case), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--rollouts", type=Path)
    parser.add_argument("--saved-inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=512)
    args = parser.parse_args()
    if args.saved_inputs:
        replay_saved(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
