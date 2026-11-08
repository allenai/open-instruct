"""GPU correctness and fixed-workload measurements for policy-scoped caching.

Run inside the pinned MILES image. Separate processes isolate each engine arm.
This harness uses existing engine APIs; it never patches SGLang.
"""

import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path

import sglang as sgl
import torch
from olmo_core.nn.hf import config as hf_config_utils
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from olmo_sglang import register
from sglang.srt.managers import io_struct
from transformers import AutoModelForCausalLM, AutoTokenizer


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checked(value):
    if isinstance(value, tuple):
        assert value[0], value
    elif isinstance(value, dict):
        assert value.get("success", True), value
    return value


def compact(response):
    meta = response["meta_info"]
    records = meta["output_token_logprobs"]
    assert all(math.isfinite(r[0]) for r in records), "Nonfinite behavior logprob"
    assert [r[1] for r in records] == response["output_ids"]
    return dict(
        tokens=response["output_ids"],
        logprobs=[r[0] for r in records],
        cached_tokens=meta.get("cached_tokens"),
        prompt_tokens=meta.get("prompt_tokens"),
        retractions=meta.get("num_retractions"),
        spans=meta.get("weight_versions"),
        version=meta.get("weight_version"),
    )


def compare(a, b):
    same = a["tokens"] == b["tokens"]
    drift = max((abs(x - y) for x, y in zip(a["logprobs"], b["logprobs"], strict=True)), default=0) if same else None
    return dict(same_tokens=same, max_logprob_error=drift, passed=same and drift <= 1e-4)


def prepare_tiny(root):
    hf_config_utils._register_olmo3moe_auto_classes()
    torch.manual_seed(17)
    config = Olmo3MoeConfig(
        vocab_size=512,
        hidden_size=128,
        attention_hidden_size=128,
        head_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        moe_intermediate_size=256,
        n_routed_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=128,
        dense_layers_indices=[],
        layer_types=["linear_attention", "full_attention"],
        latent_moe_dim=64,
        use_head_qk_norm=True,
        use_peri_ln=True,
        use_rope=False,
        attention_gate_type="elementwise",
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        max_position_embeddings=4096,
        sliding_window=4096,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("A_log", "dt_bias")):
                parameter.zero_()
    model.save_pretrained(root / "model")
    return model


def engine(model, cached, tiny=False, native=False):
    register()
    args = dict(
        model_path=str(model),
        skip_tokenizer_init=True,
        trust_remote_code=True,
        dtype="bfloat16",
        disable_radix_cache=not cached,
        mamba_radix_cache_strategy="extra_buffer",
        weight_version="0",
        cuda_graph_backend_decode="full",
        cuda_graph_max_bs_decode=16,
        cuda_graph_backend_prefill="disabled",
        max_running_requests=16,
        max_mamba_cache_size=128,
        schedule_policy="lpm",
        stream_interval=1,
        context_length=4096 if tiny else 65536,
        max_total_tokens=32768 if tiny else 1048576,
        chunked_prefill_size=256 if tiny else 4096,
        mem_fraction_static=0.15 if tiny else 0.65,
    )
    if tiny:
        args["attention_backend"] = "torch_native"
    if cached and not native:
        args["radix_cache_backend"] = "olmo_policy_versioned_v1"
    return sgl.Engine(**args)


def generate(worker, prompts, tokens=256, salt=None):
    return worker.generate(
        input_ids=prompts,
        cache_salt=salt,
        return_logprob=True,
        sampling_params=dict(temperature=0, max_new_tokens=tokens, ignore_eos=True),
    )


def publish(worker, tensors, version):
    loop, tm = worker.loop, worker.tokenizer_manager
    loop.run_until_complete(tm.pause_generation(io_struct.PauseGenerationReqInput(mode="in_place")))
    checked(worker.begin_weight_update())
    checked(worker.update_weights_from_tensor(tensors, flush_cache=False))
    checked(worker.end_weight_update())
    loop.run_until_complete(
        tm.update_weight_version(
            io_struct.UpdateWeightVersionReqInput(new_version=str(version), abort_all_requests=False)
        )
    )
    loop.run_until_complete(tm.continue_generation(io_struct.ContinueGenerationReqInput()))


def live_response(worker, prompt, tensors, version):
    latest = {}

    async def consume():
        stream = await worker.async_generate(
            input_ids=prompt,
            return_logprob=True,
            stream=True,
            sampling_params=dict(temperature=0, max_new_tokens=512, ignore_eos=True),
        )
        async for item in stream:
            latest["response"] = item
        return latest["response"]

    task = worker.loop.create_task(consume())

    async def wait_tokens():
        while len(latest.get("response", {}).get("output_ids", [])) < 64:
            if task.done():
                raise AssertionError("Generation ended before the intended live publication")
            await asyncio.sleep(0.001)

    worker.loop.run_until_complete(asyncio.wait_for(wait_tokens(), 120))
    publish(worker, tensors, version)
    result = compact(worker.loop.run_until_complete(asyncio.wait_for(task, 180)))
    assert {str(version - 1), str(version)} <= {str(s["version"]) for s in result["spans"]}
    assert not result["retractions"], result
    return result


def tiny(root, cached, native=False):
    root.mkdir(parents=True, exist_ok=True)
    model = prepare_tiny(root)
    prompt = [5 + i % 30 for i in range(1024)]
    report = {"cached": cached, "rounds": []}
    worker = engine(root / "model", cached, tiny=True, native=native)
    tensors = [(name, tensor.cuda().contiguous()) for name, tensor in model.state_dict().items()]
    try:
        report["cold"] = compact(generate(worker, prompt, 512))
        report["warm"] = compact(generate(worker, prompt, 512))
        assert compare(report["cold"], report["warm"])["passed"]
        if cached:
            assert report["warm"]["cached_tokens"] > 0
        # Later hits can reuse a deeper recurrent checkpoint than the first hit.
        # Compare publication against that same-boundary control, not against a
        # cold prefill with a different BF16 execution partition.
        report["cached_control"] = compact(generate(worker, prompt, 512))
        report["cached_control_repeat"] = compact(generate(worker, prompt, 512))
        write(root / "report.json", report)
        assert compare(report["cached_control"], report["cached_control_repeat"])["passed"]
        if native:
            report["passed"] = True
            write(root / "report.json", report)
            return
        for version in range(1, 7):
            # Version 1 deliberately republishes identical weights; later versions
            # change attention/recurrent projections, not merely the version label.
            if version > 1:
                torch.manual_seed(100 + version)
                with torch.no_grad():
                    for name, tensor in tensors:
                        if tensor.ndim >= 2 and "embed" not in name and "lm_head" not in name:
                            tensor.add_(torch.randn_like(tensor) * 0.0002)
            live = live_response(worker, prompt, tensors, version)
            fresh = compact(generate(worker, prompt))
            warm = compact(generate(worker, prompt))
            report["rounds"].append(dict(version=version, live=live, fresh=fresh, warm=warm))
            write(root / "report.json", report)
            assert fresh["cached_tokens"] == 0, "Old-policy cache leaked into fresh request"
            assert compare(fresh, warm)["passed"]
            if cached:
                assert warm["cached_tokens"] > 0
            if version == 1:
                assert compare(live, report["cached_control"])["passed"], "Identical weights changed live continuation"
        report["passed"] = True
        write(root / "report.json", report)
    finally:
        worker.shutdown()


def benchmark(root, model, data, cached, native=False):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    with Path(data).open() as stream:
        rows = [json.loads(line) for line in stream]
    # Three unchanged prompts span the lengths used by the training workload.
    chosen = [
        min(rows, key=lambda r: abs(r["metadata"]["prompt_tokens"] - target)) for target in (12000, 24000, 40000)
    ]
    prompts = [tokenizer.encode(r["input"], add_special_tokens=False) for r in chosen]
    worker = engine(model, cached, native=native)
    report = dict(
        cached=cached,
        prompt_ids=[r["metadata"]["sample_id"] for r in chosen],
        lengths=list(map(len, prompts)),
        waves=[],
    )
    try:
        generate(worker, prompts[0], 32, salt="warmup")
        for repeat in range(3):
            for index, prompt in enumerate(prompts):
                # Unique namespace per wave measures cold siblings, not an already
                # hot prompt replayed in successive waves. All eight share it.
                start = time.monotonic()
                outputs = generate(worker, [prompt] * 8, salt=f"wave-{repeat}-{index}")
                elapsed = time.monotonic() - start
                report["waves"].append(
                    dict(repeat=repeat, index=index, seconds=elapsed, responses=[compact(o) for o in outputs])
                )
                write(root / "report.json", report)
        report["passed"] = True
        write(root / "report.json", report)
    finally:
        worker.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["tiny", "benchmark"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--native-cache", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--data")
    args = parser.parse_args()
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "olmo_sglang.models"
    if args.mode == "tiny":
        tiny(args.output, args.cached or args.native_cache, native=args.native_cache)
    else:
        benchmark(args.output, args.model, args.data, args.cached or args.native_cache, native=args.native_cache)


if __name__ == "__main__":
    main()
