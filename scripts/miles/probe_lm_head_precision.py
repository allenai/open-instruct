"""Rescore frozen SGLang trajectories with BF16 and FP32-output Core LM heads.

Both heads consume the exact same Core hidden states and unchanged BF16 weights.
FP32 means BF16-input GEMM with FP32 accumulation/output, matching the pinned
SGLang enable_fp32_lm_head path. A strict FP32-input control is also retained.
This is an inference diagnostic, not an optimizer or training qualification.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

import torch
from olmo_core import config as core_config
from olmo_core.nn import attention
from olmo_core.nn.moe.v2 import olmo3
from safetensors.torch import load_file
from scripts.miles import benchmark_core_compat as benchmark
from transformers import AutoConfig

from open_instruct.miles import fla_compat


def head_logits(hidden, weight, *, strict=False):
    """Only change projection arithmetic; preserve the stored weight values."""
    flat = hidden.reshape(-1, hidden.shape[-1])
    result = torch.mm(flat.float(), weight.float().T) if strict else torch.mm(flat, weight.T, out_dtype=torch.float32)
    return result.reshape(*hidden.shape[:-1], weight.shape[0])


def selected_scores(logits, ids, start):
    result = []
    for offset in range(start, len(ids) - 1, 128):
        chunk = logits[0, offset : min(offset + 128, len(ids) - 1)].float()
        tokens = torch.tensor(ids[offset + 1 : offset + 1 + len(chunk)], device=chunk.device)
        result.append((chunk.gather(-1, tokens[:, None]).squeeze(-1) - chunk.logsumexp(-1)).cpu())
    return torch.cat(result)


def probability_statistics(serving, core):
    values = benchmark.summarize_delta(serving, core)
    probabilities = torch.stack((serving.double().exp(), core.double().exp()))
    values["probability_pearson"] = torch.corrcoef(probabilities)[0, 1].item()
    ratios = (core.double() - serving.double()).exp()
    values["tis_upper2_fraction"] = (ratios > 2).double().mean().item()
    values["mean_abs_ratio_minus_one"] = (ratios - 1).abs().mean().item()
    return values


def score(args):
    fla_compat.install_kda_triton_compat()
    torch.backends.cuda.matmul.allow_tf32 = False
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
    projection = model.lm_head.w_out
    if projection.bias is not None or projection.weight.dtype != torch.bfloat16:
        raise ValueError("Probe requires an unbiased BF16 projection")
    weight_digest = hashlib.sha256(projection.weight.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
    captured = {}

    def capture(module, inputs):
        captured["hidden"] = inputs[0].detach()

    handle = projection.register_forward_pre_hook(capture)
    report = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "head_weight_sha256": weight_digest,
        "head_weight_dtype": str(projection.weight.dtype),
        "head_weight_shape": list(projection.weight.shape),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "config": config.to_dict(),
        "comparisons": {},
    }
    cache = {}
    with torch.no_grad():
        for path in args.serving_reports:
            serving = json.loads(path.read_text())
            if (
                serving["model"] != args.model
                or serving["samples_sha256"] != hashlib.sha256(args.samples.read_bytes()).hexdigest()
            ):
                raise ValueError("Model or prompt provenance mismatch")
            name = path.stem
            output = {"rows": [], "aggregate": {}, "by_position": {}}
            combined = collections.defaultdict(lambda: ([], []))
            positions = collections.defaultdict(lambda: ([], []))
            for kind in ["rollouts", "forced"]:
                for rollout in serving[kind]:
                    row = rows[rollout["row"]]
                    ids = row["input_ids"] + rollout["output_ids"]
                    key = tuple(ids)
                    start = len(row["input_ids"]) - 1
                    if key not in cache:
                        logits = model(torch.tensor([ids], device="cuda"))
                        bf16 = selected_scores(logits, ids, start)
                        del logits
                        hidden = captured.pop("hidden")
                        logits = head_logits(hidden, projection.weight)
                        fp32 = selected_scores(logits, ids, start)
                        # Use the same full sequence/GEMM shape for the strict control.
                        strict_logits = head_logits(hidden, projection.weight, strict=True)
                        strict = selected_scores(strict_logits, ids, start)
                        kernel_delta = (strict_logits - logits).abs()
                        control = {
                            "max_abs_logits": kernel_delta.max().item(),
                            "mean_abs_logits": kernel_delta.mean().item(),
                            "scores": benchmark.summarize_delta(fp32, strict),
                        }
                        del logits, strict_logits, hidden, kernel_delta
                        cache[key] = ({"bf16": bf16, "fp32": fp32, "strict_fp32": strict}, control)
                    scores, control = cache[key]
                    actual = torch.tensor(rollout["logprobs"])
                    detail = {
                        "kind": kind,
                        "row": rollout["row"],
                        "repeat": rollout.get("repeat"),
                        "domain": row["domain"],
                        "core_logprobs": {k: v.tolist() for k, v in scores.items()},
                        "statistics": {k: probability_statistics(actual, v) for k, v in scores.items()},
                        "fp32_output_vs_strict": control,
                    }
                    output["rows"].append(detail)
                    for precision, expected in scores.items():
                        combined[f"{kind}/{precision}"][0].append(actual)
                        combined[f"{kind}/{precision}"][1].append(expected)
                        if kind == "rollouts":
                            for begin in range(0, len(actual), 128):
                                pkey = f"{precision}/{begin + 1}-{min(begin + 128, len(actual))}"
                                positions[pkey][0].append(actual[begin : begin + 128])
                                positions[pkey][1].append(expected[begin : begin + 128])
            for key, (a, b) in combined.items():
                output["aggregate"][key] = probability_statistics(torch.cat(a), torch.cat(b))
            for key, (a, b) in positions.items():
                output["by_position"][key] = probability_statistics(torch.cat(a), torch.cat(b))
            duration = sum(b["seconds"] for b in serving["batches"])
            count = sum(b["generated_tokens"] for b in serving["batches"])
            output["throughput"] = {"tokens": count, "seconds": duration, "tokens_per_second": count / duration}
            report["comparisons"][name] = output
            benchmark.write_json(args.output, report)
            print("LM_HEAD_RESULT", name, json.dumps(output["aggregate"]), flush=True)
    handle.remove()
    after = hashlib.sha256(projection.weight.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
    if after != weight_digest:
        raise ValueError("Head weights changed during unchanged-weight probe")
    report["head_weights_unchanged"] = True
    benchmark.write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--serving-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    score(parser.parse_args())


if __name__ == "__main__":
    main()
