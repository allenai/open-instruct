"""Trace long-prefix serving/Core boundaries and a KDA projection ablation."""

import argparse
import gc
import json
from contextlib import ExitStack
from pathlib import Path

import sglang
import torch
from fla.modules import FusedRMSNormGated
from fla.modules.convolution import causal_conv1d
from fla.ops import kda
from olmo_core import config as core_config
from olmo_core.nn import attention
from olmo_core.nn.moe.v2 import olmo3
from olmo_sglang import register
from safetensors.torch import load_file
from scripts.miles import hero_core_fidelity as fidelity
from scripts.miles import hero_numerics as numerics
from sglang.srt.layers.attention.linear import gdn_backend
from torch.nn import functional as F
from transformers import AutoConfig

from open_instruct.miles import fla_compat

MODES = ["original", "split_kda", "core_conv", "core_state", "core_norm", "all_kda"]
_KERNELS_PATCHED = False


def trace_factory(config):
    global _KERNELS_PATCHED
    root = Path(config["root"])
    if not _KERNELS_PATCHED:
        original_conv = gdn_backend.causal_conv1d_fn
        original_chunk = kda.chunk_kda

        def conv(x, weight, bias=None, **kwargs):
            mode = json.loads((root / "control.json").read_text())["case"]
            saved = x.clone() if mode in {"core_conv", "all_kda"} else None
            output = original_conv(x, weight, bias, **kwargs)
            if saved is not None:
                assert not kwargs["has_initial_state"].any(), "Full-prefix diagnostic only"
                output = (
                    causal_conv1d(
                        x=saved.t().unsqueeze(0).contiguous(),
                        weight=weight.to(x.dtype),
                        bias=bias,
                        activation="silu",
                        backend="triton",
                    )[0]
                    .squeeze(0)
                    .t()
                )
            return output

        def chunk(**kwargs):
            mode = json.loads((root / "control.json").read_text())["case"]
            if mode not in {"core_state", "all_kda"}:
                return original_chunk(**kwargs)
            assert kwargs["initial_state"].count_nonzero() == 0, "Full-prefix diagnostic only"
            kwargs.update(initial_state=None, cu_seqlens=None, transpose_state_layout=False)
            output, state = original_chunk(**kwargs)
            return output, state.transpose(-1, -2).contiguous()

        gdn_backend.causal_conv1d_fn = conv
        kda.chunk_kda = chunk
        _KERNELS_PATCHED = True
    capture = numerics.trace_factory(config)
    installed = False

    def after(module, args, output):
        nonlocal installed
        if not installed and hasattr(module, "f_proj_1"):
            original = module.qkv_proj.forward
            original_norm = module.o_norm.forward
            core_norm = FusedRMSNormGated(
                module.head_v_dim,
                eps=module.o_norm.eps,
                activation="sigmoid",
                device=module.o_norm.weight.device,
                dtype=module.o_norm.weight.dtype,
            )
            core_norm.weight = module.o_norm.weight

            def normalize(value, gate):
                mode = json.loads((root / "control.json").read_text())["case"]
                return core_norm(value, gate) if mode in {"core_norm", "all_kda"} else original_norm(value, gate)

            def project(value):
                if json.loads((root / "control.json").read_text()).get("split_kda"):
                    weights = module.qkv_proj.weight.split([module.key_dim, module.key_dim, module.value_dim])
                    return torch.cat([F.linear(value, w) for w in weights], dim=-1), None
                return original(value)

            module.qkv_proj.forward = project
            module.o_norm.forward = normalize
        installed = True
        return capture(module, args, output)

    return after


def run(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    samples = json.loads(args.samples.read_text())["rows"]
    rollout = json.loads(args.rollouts.read_text())["rollouts"][0]
    ids = samples[rollout["row"]]["input_ids"] + rollout["output_ids"]
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    register()
    (root / "control.json").write_text(json.dumps({"case": "prime"}))
    targets = ["model.embed_norm", "model.norm", "logits_processor"]
    targets += [f"model.layers.{i}.{name}" for i in range(config.num_hidden_layers) for name in numerics.BOUNDARIES]
    engine = sglang.Engine(
        model_path=args.model,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tp_size=1,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        attention_backend="triton",
        sampling_backend="pytorch",
        context_length=4096,
        max_total_tokens=8192,
        max_running_requests=1,
        max_mamba_cache_size=8,
        chunked_prefill_size=8192,
        mem_fraction_static=0.65,
        skip_server_warmup=True,
        log_level="error",
        forward_hooks=[
            {
                "name": name,
                "target_modules": [name],
                "hook_factory": "scripts.miles.trace_core_compat:trace_factory",
                "config": {"root": str(root), "module": name},
            }
            for name in targets
        ],
    )
    try:
        engine.generate(input_ids=ids[:16], sampling_params={"temperature": 0, "max_new_tokens": 1})
        for mode in MODES:
            (root / "control.json").write_text(json.dumps({"case": mode, "split_kda": mode == "split_kda"}))
            result = engine.generate(
                input_ids=ids,
                sampling_params={"temperature": 0, "max_new_tokens": 1},
                return_logprob=True,
                logprob_start_len=0,
            )
            (root / f"{mode}.json").write_text(json.dumps(result))
            print("LONG_TRACE_SERVING", mode, len(ids), flush=True)
    finally:
        engine.shutdown()
    gc.collect()
    torch.cuda.empty_cache()
    fla_compat.install_kda_triton_compat()
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
        assert not state.keys() & part.keys()
        state.update(part)
    olmo3.load_olmo3_moe_hf_state(model, config, state)
    del state
    model.eval()
    with torch.no_grad(), ExitStack() as stack:
        traces = [stack.enter_context(fidelity.capture_block(b, "core")) for b in model.blocks.values()]
        logits = model(torch.tensor([ids], device="cuda"))
        reference = fidelity.summarize_logits(logits, ids)
    report = {"tokens": len(ids), "modes": {}}
    for mode in MODES:
        values = {}
        for index, trace in enumerate(traces):
            layer = {}
            for label, (sg_name, _) in fidelity.BOUNDARIES.items():
                path = root / "sglang" / mode / f"model.layers.{index}.{sg_name}.pt"
                observed = torch.load(path, weights_only=True)
                for direction in ["input", "output"]:
                    layer[f"{label}.{direction}"] = fidelity.metric(observed[direction], trace[f"{label}.{direction}"])
            values[str(index)] = layer
        result = json.loads((root / f"{mode}.json").read_text())
        observed_lp = torch.tensor([item[0] for item in result["meta_info"]["input_token_logprobs"]][1:])
        last = torch.load(root / "sglang" / mode / "logits.pt", weights_only=True)["output"].reshape(-1)
        report["modes"][mode] = {
            "layers": values,
            "selected": fidelity.metric(observed_lp, reference["selected"]),
            "last": fidelity.distribution_comparison(last, reference["last"]),
        }
        print("LONG_TRACE_RESULT", mode, report["modes"][mode]["selected"], flush=True)
    (root / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
