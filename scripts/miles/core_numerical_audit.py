"""Compare frozen Core forwards under controlled FLA kernel perturbations."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from miles.backends.core_utils import fla_compat
from olmo_core import config as core_config
from olmo_core.nn import attention
from olmo_core.nn.moe.v2 import olmo3
from olmo_sglang import kernel_profile
from safetensors.torch import load_file
from transformers import AutoConfig

from open_instruct.miles.distillation import audit_tuning, numerical_audit


def metric(actual, reference):
    actual, reference = actual.float().reshape(-1), reference.float().reshape(-1)
    if actual.shape != reference.shape:
        raise ValueError("Mismatched capture shapes")
    delta = (actual - reference).abs()
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "different_fraction": (actual != reference).float().mean().item(),
    }


def worker(spec, backend):
    root = Path(spec["output"]["root"]) / backend
    root.mkdir(parents=True, exist_ok=False)
    source = Path(spec["launch"]["env"]["OI_MILES_OPD_AUDIT_INPUT"])
    saved = json.loads(source.read_text())
    if saved["same_checkpoint"] != spec["model"]["source"]:
        raise ValueError("Mismatched checkpoint")
    reference_root = Path(spec["launch"]["env"]["OI_MILES_OPD_CORE_REFERENCE"])
    sg_records = [json.loads(line) for line in (reference_root / "scores.jsonl").read_text().splitlines()]
    hf = AutoConfig.from_pretrained(saved["same_checkpoint"], trust_remote_code=True)
    fla_compat.install_kda_triton_compat()
    torch.distributed.init_process_group(
        backend="nccl", init_method=f"file://{root / 'rendezvous'}", rank=0, world_size=1
    )
    config = olmo3.build_olmo3_moe_config_from_hf_config(
        hf,
        dtype=core_config.DType.bfloat16,
        attention_backend=attention.AttentionBackendName(backend),
        router_aux_loss_weight=0.0,
        router_z_loss_weight=0.0,
    )
    config.recompute_each_block = False
    model = config.build(init_device="meta")
    model.init_weights(max_seq_len=4096, max_local_microbatch_size=4096, device=torch.device("cuda"))
    state = {}
    start = time.monotonic()
    for shard in sorted(Path(saved["same_checkpoint"]).glob("*.safetensors")):
        part = load_file(shard)
        if state.keys() & part.keys():
            raise ValueError("Duplicate checkpoint weights")
        state.update(part)
    olmo3.load_olmo3_moe_hf_state(model, hf, state)
    del state
    for block in model.routed_blocks():
        block.routed_experts_router.set_load_balancing_process_group(torch.distributed.group.WORLD)
    model.eval()
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "checkpoint": saved["same_checkpoint"],
                "backend": backend,
                "gpu": torch.cuda.get_device_name(),
                "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "load_seconds": time.monotonic() - start,
                "torch": torch.__version__,
                "core_source_sha256": hashlib.sha256(Path(olmo3.__file__).read_bytes()).hexdigest(),
                "config": config.as_dict(),
                "diagnostic_only": True,
            },
            indent=2,
            default=str,
        )
    )
    captures = {}
    handles = []

    def capture(name, route=False, before=False):
        def hook(module, args, output=None):
            value = args[0] if before else (output[1] if route else output)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Unexpected boundary output: {name}")
            captures[name] = value.detach().cpu().clone()

        return hook

    for name, block in model.blocks.items():
        handles.append(block.register_forward_hook(capture(f"block.{name}")))
        if getattr(block, "routed_experts_router", None) is not None:
            handles.append(block.routed_experts_router.register_forward_hook(capture(f"routes.{name}", route=True)))
        if hasattr(block.attention, "o_norm"):
            handles.append(block.attention.o_norm.register_forward_pre_hook(capture(f"kda.{name}", before=True)))

    def score(label, tokens=None, training=False):
        tokens = saved["tokens"] if tokens is None else tokens
        ids = saved["prefix"] + tokens
        captures.clear()
        model.train(training)
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.set_grad_enabled(training):
            logits = model(torch.tensor([ids], device="cuda"))[0].float()
            response_logits = logits[len(saved["prefix"]) - 1 : -1]
            target = torch.tensor(tokens, device="cuda")
            scores = (
                (response_logits.gather(-1, target[:, None]).squeeze(-1) - response_logits.logsumexp(-1))
                .detach()
                .cpu()
                .tolist()
            )
            last = logits[-1].detach().cpu()
        torch.cuda.synchronize()
        numerical_audit.append(
            root / "scores.jsonl",
            {
                "label": label,
                "tokens": tokens,
                "prefix": saved["prefix"],
                "scores": scores,
                "seconds": time.monotonic() - started,
                "grad_forward": training,
            },
        )
        return scores, dict(captures), last

    tuners = {name: kernel_profile.tuner(name) for name in kernel_profile.KERNELS}
    warm, _, _ = score("autotuned-warmup")
    baseline_configs = {name: dict(tuner.cache) for name, tuner in tuners.items()}
    donor = json.loads(Path(spec["launch"]["env"]["OI_MILES_OPD_AUDIT_TUNING"]).read_text())["core-eager"]
    # The Core dispatch has different cache keys (e.g. no requested final state).
    # Match actual available launch configurations, retaining Core's observed keys.
    for name, tuner in tuners.items():
        values = set(donor[name].values())
        if len(values) != 1 or not baseline_configs[name]:
            raise ValueError(f"Expected one donor choice and observed Core keys: {name}")
        choice = next(c for c in tuner.configs if str(c) in values)
        baseline_configs[name] = dict.fromkeys(baseline_configs[name], choice)

    used = set()

    def install(configs):
        used.clear()
        for name, tuner in tuners.items():
            tuner.cache = kernel_profile.StrictCache(configs[name], name, used)

    def record(label, values, states, reference, reference_states, metadata=None):
        if {name for name, _ in used} != set(tuners):
            raise RuntimeError("A managed KDA kernel was not observed during scoring")
        layers = {}
        for name, value in states.items():
            if name.startswith("routes."):
                a, b = value.sort(-1).values, reference_states[name].sort(-1).values
                layers[name] = {"changed_token_fraction": (a != b).any(-1).float().mean().item()}
            else:
                layers[name] = metric(value, reference_states[name])
        result = {
            "label": label,
            "comparison": numerical_audit.compare(values, reference),
            "layers": layers,
            "intervention": metadata,
            "observed_kernel_keys": len(used),
        }
        numerical_audit.append(root / "comparisons.jsonl", result)
        print(json.dumps({"backend": backend, "label": label, "comparison": result["comparison"]}), flush=True)

    install(baseline_configs)
    baseline, baseline_states, last = score("matched-donor")
    torch.save({"states": baseline_states, "last_logits": last}, root / "baseline.pt")
    (root / "matched-configs.json").write_text(
        json.dumps(
            {name: {repr(key): str(c) for key, c in entries.items()} for name, entries in baseline_configs.items()},
            indent=2,
        )
    )
    numerical_audit.append(
        root / "comparisons.jsonl",
        {"label": "autotuned-vs-matched", "comparison": numerical_audit.compare(warm, baseline)},
    )
    for record_sg in sg_records:
        if record_sg["label"] == "cold-0-0":
            if record_sg["tokens"] != saved["tokens"] or record_sg["prefix"] != saved["prefix"]:
                raise ValueError("SGLang token mismatch")
            numerical_audit.append(
                root / "comparisons.jsonl",
                {
                    "label": f"sglang-{record_sg['mode']}",
                    "comparison": numerical_audit.compare(baseline, record_sg["scores"]),
                },
            )
    for label, training in (("matched-repeat", False),):
        values, states, _ = score(label, training=training)
        record(label, values, states, baseline, baseline_states)
    for name, tuner in tuners.items():
        original = next(iter(baseline_configs[name].values()))
        for kind, chosen in audit_tuning.alternatives(tuner.configs, original):
            configs = {n: dict(entries) for n, entries in baseline_configs.items()}
            configs[name] = dict.fromkeys(configs[name], chosen)
            install(configs)
            label = name.rsplit(".", 1)[-1] + "-" + kind
            values, states, _ = score(label)
            record(
                label,
                values,
                states,
                baseline,
                baseline_states,
                {"kernel": name, "original": str(original), "changed": str(chosen)},
            )
    install(baseline_configs)
    values, states, _ = score("restored-repeat")
    record("restored-repeat", values, states, baseline, baseline_states)
    values, states, _ = score("grad-forward", training=True)
    record("grad-forward", values, states, baseline, baseline_states)
    training_scores, training_states = values, states
    # Match MILES prepare_model_config; the unmodified Core default above is
    # deliberately retained as a numerical-sensitivity control.
    for block in model.blocks.values():
        experts = getattr(block, "routed_experts", None)
        if experts is not None:
            experts.match_eager_rounding = True
            experts.row_specialization = "dynamic"
    for label, training in (("miles-eval-rounded", False), ("miles-train-rounded", True)):
        values, states, _ = score(label, training=training)
        record(label, values, states, training_scores, training_states)
    for record_sg in sg_records:
        if record_sg["label"] == "decode":
            values, _, _ = score("core-rescore-" + record_sg["mode"], record_sg["tokens"])
            for other in sg_records:
                if other["mode"] == record_sg["mode"] and other["label"] in ("decode", "generated-prefill-0"):
                    numerical_audit.append(
                        root / "comparisons.jsonl",
                        {
                            "label": "core-vs-" + other["mode"] + "-" + other["label"],
                            "comparison": numerical_audit.compare(values, other["scores"]),
                        },
                    )
    for handle in handles:
        handle.remove()
    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--backend", choices=["torch", "flash_4"])
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.backend:
        worker(spec, args.backend)
        return
    root = Path(spec["output"]["root"])
    root.mkdir(parents=True, exist_ok=False)
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    if len(devices) != 2:
        raise ValueError("Expected two GPUs for independent Core attention-backend checks")
    processes = []
    for device, backend in zip(devices, ["torch", "flash_4"], strict=True):
        processes.append(
            subprocess.Popen(
                [sys.executable, __file__, str(args.spec), "--backend", backend],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": device},
            )
        )
    codes = [p.wait() for p in processes]
    if any(codes):
        raise RuntimeError(f"Core audit workers failed: {codes}")


if __name__ == "__main__":
    main()
