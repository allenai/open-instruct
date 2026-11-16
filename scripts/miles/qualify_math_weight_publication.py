"""Repeated expert publication gate using Core's pinned flattened NCCL transport.

Run on the study's four B300 layout: sender GPU0, TP1 servers GPU1..3.
This isolates transport/loading from optimization; the separate training gate
must still qualify Core export, optimizer steps, checkpoint and resume.
"""

import argparse
import hashlib
import importlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

import httpx
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from scripts.miles import diagnostic_native_config, qualify_math_nvfp4_kv, qualify_math_quantization
from torch import distributed as dist

from open_instruct.miles.execution import fp8_publication, kda_dispatch, quantized_publication

EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
# Preregistered tolerances on fixed-token conditional log probabilities. Full
# parameter hashes must ALSO match fresh loading, including every quant scale.
TOLERANCES = {"bf16": (2e-4, 1e-4), "nvfp4": (0.05, 0.01)}


def configure_transport_environment():
    # Match miles.ray.specs.train.compute_trainer_env_vars. Without this,
    # the standalone sender uses NCCL's cuMem default while SGLang explicitly
    # disables it in _set_envs_and_config, which can break their first collective.
    # Set before creating any process group; children inherit the same value.
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")


def instrument_receiver():
    """Retain the CUDA context and original traceback if transport fails."""
    updater_type = importlib.import_module(
        "sglang.srt.model_executor.model_runner_components.weight_updater"
    ).WeightUpdater
    original = updater_type.receive_weights_from_distributed

    def receive(updater, **options):
        details = {
            "phase": "receive",
            "pid": os.getpid(),
            "gpu_id": updater.gpu_id,
            "current_device": torch.cuda.current_device(),
            "device": updater.device,
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE"),
            "tensors": len(options["names"]),
            "first": options["names"][0],
        }
        context = Path(os.environ["OI_PUBLICATION_AUDIT"]) / "transport-context.json"
        if not context.exists():
            save(context, details)
        try:
            return original(updater, **options)
        except Exception:
            details["traceback"] = traceback.format_exc()
            save(Path(os.environ["OI_PUBLICATION_AUDIT"]) / "transport-error.json", details)
            print(json.dumps(details), flush=True)
            raise

    updater_type.receive_weights_from_distributed = receive


def make_probes(tokenizer):
    probes = []
    for n in range(16):
        content = (
            f"Compute {n} times {n + 7}. Explain each arithmetic step, check your calculation using addition, "
            "and give the final answer within \\boxed{}. This is a deterministic weight publication diagnostic."
        )
        # The pinned Transformers returns BatchEncoding from tokenize=True.
        # Render the exact native template, then explicitly request token IDs.
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 32 or not all(isinstance(token, int) for token in ids):
            raise ValueError("Invalid fixed-token probe encoding")
        probes.append(ids)
    return probes


def full_cap_probes(tokenizer, path, expected_hash):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("Full-cap training prompt snapshot hash mismatch")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    probes = []
    for row in rows:
        ids = tokenizer.encode(row["input"], add_special_tokens=False)
        if ids != row["metadata"]["prompt_token_ids"]:
            raise ValueError("Full-cap prompt/tokenizer mismatch")
        if not 32 <= len(ids) <= 1465 or len(ids) + 16384 + 2 > 18432:
            raise ValueError("Full-cap prompt exceeds approved RL context")
        probes.append(ids)
    if len(probes) < 16 or max(map(len, probes)) != 1465:
        raise ValueError("Full-cap snapshot must cover the actual 1465-token maximum")
    return sorted(probes, key=len, reverse=True)[:16]


def verify_full_cap_outputs(outputs, count):
    if not isinstance(outputs, list) or len(outputs) != count:
        raise ValueError("Missing full-cap responses")
    for output in outputs:
        meta = output["meta_info"]
        scores = meta.get("output_token_logprobs", [])
        candidates = meta.get("output_top_logprobs", [])
        if meta.get("completion_tokens") != 16384 or len(scores) != 16384 or len(candidates) != 16384:
            raise ValueError("Full-cap output must contain exactly 16384 scored tokens")
        if any(len(step) != 20 for step in candidates):
            raise ValueError("Missing full-cap candidate20 coverage")
        values = [entry[0] for entry in scores] + [entry[0] for step in candidates for entry in step]
        if not torch.isfinite(torch.tensor(values)).all():
            raise ValueError("Nonfinite full-cap logprobs/candidates")


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def changed_tensor(name, tensor, variant):
    """Deterministic BF16 expert-only changes, reproducible in a fresh checkpoint."""
    match = EXPERT.fullmatch(name)
    if variant == 0 or match is None:
        return tensor
    # Change all routed experts, including both halves of fused w13. Dense,
    # attention, routers, shared experts, head and recurrent parameters stay put.
    factor = (0.875, 1.125)[variant - 1]
    return (tensor.float() * factor).to(torch.bfloat16)


def weight_buckets(reader, variant, max_bytes):
    """HF per-expert names; force w1/w3 into distinct buckets at expert0/layer."""
    bucket, size = [], 0
    for name in sorted(reader.keys()):
        tensor = changed_tensor(name, reader.get_tensor(name), variant)
        match = EXPERT.fullmatch(name)
        force = match is not None and int(match[2]) == 0
        if bucket and (force or size + tensor.nbytes > max_bytes):
            yield bucket
            bucket, size = [], 0
        bucket.append((name, tensor))
        size += tensor.nbytes
        if force:
            yield bucket
            bucket, size = [], 0
    if bucket:
        yield bucket


def compare_scores(actual, expected, precision, *, unchanged=False):
    if len(actual) != len(expected) or not actual:
        raise ValueError("Missing fixed-token probes")
    differences, references = [], []
    for left, right in zip(actual, expected, strict=True):
        if unchanged and (left.get("text"), left["meta_info"].get("output_token_logprobs")) != (
            right.get("text"),
            right["meta_info"].get("output_token_logprobs"),
        ):
            raise ValueError("Unchanged publication changed generated outputs")
        a, b = left["meta_info"]["input_token_logprobs"], right["meta_info"]["input_token_logprobs"]
        if len(a) != len(b) or len(a) < 32:
            raise ValueError("Incomplete fixed-token scores")
        for x, y in zip(a, b, strict=True):
            if x[1] != y[1] or (x[0] is None) != (y[0] is None):
                raise ValueError("Probe token alignment changed")
            if x[0] is not None:
                differences.append(float(x[0]) - float(y[0]))
                references.append(float(y[0]))
    diff, ref = torch.tensor(differences), torch.tensor(references)
    if not diff.numel() or not torch.isfinite(diff).all() or not torch.isfinite(ref).all():
        raise ValueError("Nonfinite or missing scores")
    if precision == "fp8" and not unchanged:
        return {
            "tokens": diff.numel(),
            "max_abs": diff.abs().max().item(),
            "numerical_equivalence_asserted": False,
            "scope": "Finite aligned scores only; exact parameter/scale/layout parity checked independently",
        }
    atol, rtol = (0.0, 0.0) if unchanged else TOLERANCES[precision]
    if not torch.all(diff.abs() <= atol + rtol * ref.abs()):
        raise ValueError(f"Publication/fresh score mismatch: max={diff.abs().max().item()}, atol={atol}, rtol={rtol}")
    return {"tokens": diff.numel(), "max_abs": diff.abs().max().item(), "atol": atol, "rtol": rtol}


def tensor_fingerprint(tensor):
    cpu = tensor.detach().contiguous().cpu()
    return {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "sha256": hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def instrument_nvfp4_quantization():
    """Locate a mismatch before or after encoding without replacing the encoder."""
    method = importlib.import_module("sglang.srt.layers.quantization.nvfp4_online").ModelOptNvFp4OnlineFusedMoEMethod
    original = method._quantize_weight_nvfp4
    calls = 0

    def quantize(weight, weight_scale_2=None):
        nonlocal calls
        index = calls
        calls += 1
        # The approved model has 512 experts and two encodings per expert:
        # w2 and the combined w1/w3. Sample four calls per layer/publication.
        sampled = index % 1024 < 4
        before = tensor_fingerprint(weight) if sampled else None
        result = original(weight, weight_scale_2)
        if sampled:
            save(
                Path(os.environ["OI_PUBLICATION_AUDIT"]) / f"encoding-{os.getpid()}-{index}.json",
                {
                    "call": index,
                    "input": before,
                    "input_stride": list(weight.stride()),
                    "input_device": str(weight.device),
                    "default_dtype": str(torch.get_default_dtype()),
                    "provided_scale": None if weight_scale_2 is None else tensor_fingerprint(weight_scale_2),
                    "encoded": [tensor_fingerprint(value) for value in result],
                },
            )
        return result

    method._quantize_weight_nvfp4 = staticmethod(quantize)


def instrument_nvfp4_processing():
    """Observe packing boundaries without changing weights or loader behavior."""
    method = importlib.import_module("sglang.srt.layers.quantization.nvfp4_online").ModelOptNvFp4OnlineFusedMoEMethod
    original = method.process_weights_after_loading

    def process(quantizer, layer):
        epoch = getattr(layer, "_publication_packing_epoch", 0) + 1
        layer._publication_packing_epoch = epoch

        # Only two experts per layer are copied, retaining useful causal evidence
        # without copying the whole model at every packing boundary.
        def sample():
            return {
                name: tensor_fingerprint(value[:2] if value.ndim else value)
                for name, value in layer.named_parameters(recurse=False)
            }

        before = sample()
        result = original(quantizer, layer)
        directory = Path(os.environ["OI_PUBLICATION_AUDIT"])
        # Object identity is stable within a receiver and separates its layers.
        save(
            directory / f"packing-{os.getpid()}-{id(layer)}-{epoch}.json",
            {"epoch": epoch, "before": before, "after": sample(), "sample_experts": [0, 1]},
        )
        return result

    method.process_weights_after_loading = process


def fla_dispatch_snapshot():
    """Record selected native autotune configurations, without changing them."""

    def config_details(config):
        # Triton Config.__repr__ only exposes a process-local memory address.
        # Preserve the actual launch choices for comparison across receivers.
        if hasattr(config, "kwargs"):
            return {
                name: getattr(config, name)
                for name in ("kwargs", "num_warps", "num_stages", "num_ctas", "maxnreg", "enable_fp_fusion")
                if hasattr(config, name)
            }
        return repr(config)

    dispatch = {}
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith(("fla.", "olmo_sglang.kda.")):
            continue
        for attribute, value in vars(module).copy().items():
            seen = set()
            path = f"{name}.{attribute}"
            # FLA wraps autotuners in Triton heuristics; their outer object has
            # no cache. Inspect the native wrapper chain without invoking it.
            while id(value) not in seen:
                seen.add(id(value))
                cache = getattr(value, "cache", None)
                if isinstance(cache, dict) and hasattr(value, "configs"):
                    dispatch[path] = {
                        "cache": {repr(key): config_details(config) for key, config in cache.copy().items()},
                        "best_config": config_details(getattr(value, "best_config", None)),
                    }
                if not hasattr(value, "fn"):
                    break
                value = value.fn
                path += ".fn"
    return dispatch


def instrument_kda_inputs():
    """Retain selected real prefill inputs; never change native cache state."""
    if os.environ.get("OI_PUBLICATION_TRACE_KDA") != "1":
        return
    kernel = importlib.import_module("olmo_sglang.kda.backend").OlmoFLAKDAKernel
    original = kernel._run
    capture_indices = {int(index) for index in os.environ.get("OI_PUBLICATION_KDA_CAPTURE_INDEX", "0").split(",")}
    if any(index < 0 for index in capture_indices):
        raise ValueError("KDA capture index must be nonnegative")
    captured = set()
    prefill_index = -1

    def run(instance, *values, **options):
        nonlocal captured, prefill_index
        if captured == capture_indices or not values or values[0].shape[1] < 32:
            return original(instance, *values, **options)
        prefill_index += 1
        if prefill_index not in capture_indices:
            return original(instance, *values, **options)
        captured.add(prefill_index)
        # Snapshot only referenced states, before the native call commits them.
        # Detached CPU copies synchronize and are explicitly diagnostic only.
        initial, valid = instance._gather_initial_state(options["ssm_states"], options["cache_indices"])
        inputs = {
            "positional": [value.detach().cpu().clone() for value in values],
            "options": {
                name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                for name, value in options.items()
                if name not in {"ssm_states", "track_state"}
            },
            "initial_state": initial.detach().cpu().clone(),
            "valid": valid.detach().cpu().clone(),
        }
        directory = Path(os.environ["OI_PUBLICATION_AUDIT"])
        if len(capture_indices) > 1:
            directory = directory / f"recurrence-{prefill_index}"
            directory.mkdir(parents=True, exist_ok=False)
        torch.save(inputs, directory / f"kda-inputs-{os.getpid()}.pt")
        result = original(instance, *values, **options)
        output = result[0] if isinstance(result, tuple) else result
        torch.save(output.detach().cpu().clone(), directory / f"kda-output-{os.getpid()}.pt")
        save(
            directory / f"kda-receipt-{os.getpid()}.json",
            {
                "diagnostic_only": True,
                "synchronizes_cuda": True,
                "cache_modified_by_instrumentation": False,
                "prefill_recurrence_index": prefill_index,
                "core_compat": getattr(instance, "core_compat", None),
                "allow_neg_eigval": getattr(instance, "allow_neg_eigval", None),
                "fla_dispatch": fla_dispatch_snapshot(),
            },
        )
        return result

    kernel._run = run


def trace_forward(model, original_forward, values, options):
    """One prefill trace for locating divergence; never a qualification receipt.

    CPU fingerprints synchronize CUDA and can hide timing bugs. A repaired gate
    must therefore pass again without this explicitly enabled instrumentation.
    """

    def tensors(value):
        if isinstance(value, torch.Tensor):
            return {**tensor_fingerprint(value), "stride": list(value.stride())}
        if isinstance(value, (tuple, list)):
            return [tensors(item) for item in value]
        if isinstance(value, dict):
            return {key: tensors(item) for key, item in value.items()}
        return None

    records, handles = [], []

    def observe(name, module, inputs, output):
        records.append({"module": name, "input": tensors(inputs), "output": tensors(output)})

    buffers = {name: tensors(value) for name, value in model.named_buffers()}
    try:
        for name, module in model.named_modules():
            if module is not model:
                handles.append(module.register_forward_hook(lambda m, i, o, name=name: observe(name, m, i, o)))
        result = original_forward(model, *values, **options)
    finally:
        for handle in handles:
            handle.remove()
    save(
        Path(os.environ["OI_PUBLICATION_AUDIT"]) / f"prefill-trace-{os.getpid()}.json",
        {
            "diagnostic_only": True,
            "synchronizes_cuda": True,
            "input": tensors(values),
            "buffers": buffers,
            "modules": records,
        },
    )
    model._publication_prefill_traced = True
    return result


def audit_scheduler(*args, **kwargs):
    """Observe real serving weights after each load, without changing numerics."""
    kda_dispatch.install(
        server_args=args[0] if args else kwargs.get("server_args"),
        gpu_id=args[2] if len(args) > 2 else kwargs.get("gpu_id"),
    )
    diagnostic = os.environ.get("OI_MILES_PUBLICATION_DIAGNOSTICS") == "1"
    if diagnostic:
        instrument_receiver()
    if os.environ.get("OI_PUBLICATION_TRACE_ENCODING") == "1":
        instrument_nvfp4_quantization()
    if os.environ.get("OI_PUBLICATION_TRACE_PACKING") == "1":
        instrument_nvfp4_processing()
    intervention = os.environ.get("OI_PUBLICATION_NATIVE_CONFIG_RECEIPT")
    if intervention:
        if os.environ.get("OI_MILES_PUBLICATION_DIAGNOSTICS") != "1":
            raise ValueError("Native intervention requires isolated diagnostic mode")
        diagnostic_native_config.install(intervention, Path(os.environ["OI_PUBLICATION_AUDIT"]), save)
    instrument_kda_inputs()
    model_type = importlib.import_module("olmo_sglang.models.olmo3_moe").Olmo3MoeForCausalLM
    quantized_publication.install_nvfp4_publication_context(model_type)
    if getattr(args[0] if args else kwargs.get("server_args"), "quantization", None) == "fp8":
        fp8_publication.install_fp8_publication()
    original_load, original_forward = model_type.load_weights, model_type.forward

    def load(model, *values, **options):
        incoming = {}

        def observed_weights():
            for item in values[0]:
                name, tensor, *_ = item
                match = EXPERT.fullmatch(name)
                if match is not None and int(match[2]) < 2:
                    incoming[name] = tensor_fingerprint(tensor)
                yield item

        result = original_load(model, observed_weights() if diagnostic else values[0], *values[1:], **options)
        model._publication_audit_epoch = getattr(model, "_publication_audit_epoch", 0) + 1
        if incoming:
            params = dict(model.named_parameters())
            resolver = getattr(model, "_weight_targets", None)
            stale = (
                []
                if resolver is None
                else [name for name, value in resolver.params.items() if params.get(name) is not value]
            )
            save(
                Path(os.environ["OI_PUBLICATION_AUDIT"])
                / f"incoming-{os.getpid()}-{model._publication_audit_epoch}.json",
                {"weights": incoming, "stale_cached_parameters": stale},
            )
        return result

    def forward(model, *values, **options):
        # Strict gates compute outputs before any hash/copy/synchronization.
        # Diagnostic traces retain their historical ordering only when opted in.
        result = None if diagnostic else original_forward(model, *values, **options)
        epoch = getattr(model, "_publication_audit_epoch", 0)
        if getattr(model, "_publication_audited_epoch", None) != epoch:
            if getattr(model, "quant_config", None) is not None and model.quant_config.get_name() == "fp8":
                storage = {name: tensor.data_ptr() for name, tensor in model.named_parameters()}
                previous = getattr(model, "_fp8_gate_storage", storage)
                if previous != storage:
                    raise ValueError("FP8 publication changed graph-visible parameter addresses")
                model._fp8_gate_storage = storage
                save(
                    Path(os.environ["OI_PUBLICATION_AUDIT"]) / f"storage-{epoch}.json",
                    {
                        "pointers": storage,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "allocated_bytes": torch.cuda.memory_allocated(),
                    },
                )
            inventory = []
            for name, module in model.named_modules():
                method = getattr(module, "quant_method", None)
                direct_parameters = list(module.named_parameters(recurse=False))
                if method is None and not direct_parameters:
                    continue
                parameters = {}
                for key, tensor in direct_parameters:
                    cpu = tensor.detach().contiguous().cpu()
                    if cpu.is_floating_point() and not torch.isfinite(cpu.float()).all():
                        raise ValueError(f"Nonfinite serving tensor {name}.{key}")
                    digest = hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                    parameters[key] = {"dtype": str(cpu.dtype), "shape": list(cpu.shape), "sha256": digest}
                inventory.append(
                    {
                        "name": name,
                        "class": type(module).__name__,
                        "quant_method": type(method).__name__,
                        "parameters": parameters,
                    }
                )
            save(Path(os.environ["OI_PUBLICATION_AUDIT"]) / f"audit-{os.getpid()}-{epoch}.json", inventory)
            model._publication_audited_epoch = epoch
        input_ids = values[0] if values else options.get("input_ids")
        if (
            os.environ.get("OI_PUBLICATION_TRACE_PREFILL") == "1"
            and not getattr(model, "_publication_prefill_traced", False)
            and isinstance(input_ids, torch.Tensor)
            and input_ids.numel() >= 32
        ):
            return trace_forward(model, original_forward, values, options)
        return original_forward(model, *values, **options) if diagnostic else result

    model_type.load_weights, model_type.forward = load, forward
    scheduler = importlib.import_module("sglang.srt.managers.scheduler")
    if getattr(args[0] if args else kwargs.get("server_args"), "kv_cache_dtype", None) == "nvfp4":
        qualify_math_nvfp4_kv.install_scheduler_audit(scheduler.Scheduler)
    return scheduler.run_scheduler_process(*args, **kwargs)


def server_options(args):
    options = dict(
        model_path=str(args.model),
        host="127.0.0.1",
        port=args.port,
        base_gpu_id=args.gpu,
        trust_remote_code=True,
        dtype="bfloat16",
        tp_size=1,
        context_length=18432,
        max_running_requests=16,
        disable_radix_cache=True,
        max_total_tokens=393216,
        max_mamba_cache_size=128,
        mem_fraction_static=0.7,
        cuda_graph_backend_decode="full",
        cuda_graph_max_bs_decode=64,
        cuda_graph_backend_prefill="disabled",
        chunked_prefill_size=4096,
        attention_backend="triton",
        sampling_backend="pytorch",
        random_seed=17,
        enable_fp32_lm_head=True,
        kv_cache_dtype="bf16",
    )
    if args.precision == "fp8":
        options.update(quantization="fp8", moe_runner_backend="triton")
    if args.precision == "nvfp4":
        options.update(quantization="nvfp4_online", moe_runner_backend="flashinfer_trtllm_routed")
    if getattr(args, "nvfp4_kv", False):
        if args.precision != "nvfp4":
            raise ValueError("Native NVFP4 KV gate requires the selected NVFP4 expert arm")
        options.update(
            kv_cache_dtype="nvfp4",
            prefill_kv_cache_dequant_dtype="nvfp4",
            page_size=16,
            attention_backend="trtllm_mha",
        )
    return options


def server(args):
    os.environ["OLMO_SGLANG_CORE_COMPAT"] = "auto" if args.precision == "bf16" else "off"
    if args.precision == "fp8":
        os.environ["SGLANG_FP8_IGNORED_LAYERS"] = ",".join(qualify_math_quantization.FP8_EXCLUSIONS)
    importlib.import_module("olmo_sglang").register()
    server_args = importlib.import_module("sglang.srt.server_args").ServerArgs
    options = server_options(args)
    os.environ["OI_KV_GATE_OUTPUT"] = str(args.output)
    save(args.output / "server-options.json", options)
    importlib.import_module("sglang.srt.entrypoints.http_server").launch_server(
        server_args(**options), run_scheduler_process_func=audit_scheduler
    )


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Fleet:
    def __init__(self, args, model, label):
        self.processes, self.clients, self.directories, self.streams = [], [], [], []
        self.args, self.model, self.label = args, model, label

    def __enter__(self):
        client_type = importlib.import_module("miles.backends.sglang_utils.sglang_api_client").SGLangApiClient
        try:
            for gpu in (1, 2, 3):
                destination = self.args.output / self.label / f"gpu{gpu}"
                destination.mkdir(parents=True)
                number = port()
                env = dict(os.environ, OI_PUBLICATION_AUDIT=str(destination))
                env["NCCL_DEBUG_FILE"] = str(destination / "nccl-%h-%p.log")
                stream = (destination / "server.log").open("w")
                self.streams.append(stream)
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "scripts.miles.qualify_math_weight_publication",
                        "--server",
                        "--model",
                        str(self.model),
                        "--precision",
                        self.args.precision,
                        "--output",
                        str(destination),
                        "--gpu",
                        str(gpu),
                        "--port",
                        str(number),
                        *(["--nvfp4-kv"] if getattr(self.args, "nvfp4_kv", False) else []),
                    ],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self.processes.append(process)
                self.clients.append(client_type(f"http://127.0.0.1:{number}"))
                self.directories.append(destination)
            deadline = time.monotonic() + 1800
            with httpx.Client(timeout=5) as client:
                pending = set(range(3))
                while pending:
                    for index in list(pending):
                        if self.processes[index].poll() is not None:
                            raise RuntimeError(f"Server exited: {self.directories[index]}")
                        try:
                            response = client.get(self.clients[index].server_url + "/health")
                            if response.status_code == 200:
                                pending.remove(index)
                        except httpx.HTTPError:
                            pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("Fresh server readiness deadline")
                    if pending:
                        time.sleep(2)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for process in self.processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in self.processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for stream in self.streams:
            stream.close()

    def probe(self, label, probes):
        if getattr(self.args, "full_cap_prompts", None):
            return self.full_cap_probe(label, probes)
        result, audits = [], []
        with httpx.Client(timeout=600) as client:
            for engine, directory in zip(self.clients, self.directories, strict=True):
                response = client.post(
                    engine.server_url + "/generate",
                    json={
                        "input_ids": probes,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 4 if self.args.precision == "fp8" else 1,
                        },
                        "return_logprob": True,
                        "logprob_start_len": 0,
                        "top_logprobs_num": 20,
                    },
                )
                response.raise_for_status()
                outputs = response.json()
                if not isinstance(outputs, list) or len(outputs) != len(probes):
                    raise ValueError("Missing probe responses")
                for output in outputs:
                    candidates = output["meta_info"].get("output_top_logprobs")
                    if not candidates or len(candidates[0]) != 20:
                        raise ValueError("Missing top20 candidates")
                result.append(outputs)
                paths = sorted(directory.glob("audit-*.json"), key=lambda p: p.stat().st_mtime_ns)
                if not paths:
                    raise ValueError("Missing serving tensor audit")
                audit = json.loads(paths[-1].read_text())
                if getattr(self.args, "nvfp4_kv", False):
                    pools = list(directory.glob("pool-audit-*.json"))
                    if len(pools) != 1:
                        raise ValueError("Missing unique actual native KV/recurrent/backend audit")
                    qualify_math_nvfp4_kv.verify_audit(json.loads(pools[0].read_text()))
                    # KV scale loaders are validated against actual physical pools
                    # above; retain their hashes in the complete comparison audit.
                    weights = [module for module in audit if module["class"] != "RadixAttention"]
                    qualify_math_quantization.verify_inventory(weights, self.args.precision)
                else:
                    qualify_math_quantization.verify_inventory(audit, self.args.precision, kv_cache_dtype="bf16")
                audits.append(audit)
        save(self.args.output / f"{label}.json", {"outputs": result, "audits": audits})
        return result, audits

    def full_cap_probe(self, label, probes):
        # All three receivers run concurrently on their own approved GPUs. This
        # is a correctness gate, never an inference throughput measurement.
        references = self.args.full_cap_conditional_inputs

        def receiver(index):
            outputs, cases = [], []
            with httpx.Client(timeout=1800) as client:
                for count in (1, 8, 16):
                    payload = {
                        "input_ids": probes[:count],
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 16384,
                            "min_new_tokens": 16384,
                            "ignore_eos": True,
                        },
                        "return_logprob": True,
                        "logprob_start_len": 0,
                        "top_logprobs_num": 20,
                    }
                    response = client.post(self.clients[index].server_url + "/generate", json=payload)
                    response.raise_for_status()
                    current = response.json()
                    verify_full_cap_outputs(current, count)
                    key = (index, count)
                    if key not in references:
                        if label != "initial":
                            raise ValueError("Missing initial full-cap conditional tokens")
                        references[key] = [
                            prefix + [entry[1] for entry in output["meta_info"]["output_token_logprobs"]]
                            for prefix, output in zip(probes[:count], current, strict=True)
                        ]
                    # Score the SAME baseline continuation after every changed,
                    # repeated and restored publication and fresh server load.
                    forced = client.post(
                        self.clients[index].server_url + "/generate",
                        json={
                            "input_ids": references[key],
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": 4 if self.args.precision == "fp8" else 1,
                            },
                            "return_logprob": True,
                            "logprob_start_len": 0,
                        },
                    )
                    forced.raise_for_status()
                    conditional = forced.json()
                    if not isinstance(conditional, list) or len(conditional) != count:
                        raise ValueError("Missing full-cap fixed-conditional scores")
                    for output, scored, tokens in zip(current, conditional, references[key], strict=True):
                        scores = scored["meta_info"]["input_token_logprobs"]
                        if len(scores) != len(tokens) or [entry[1] for entry in scores] != tokens:
                            raise ValueError("Full-cap conditional token alignment mismatch")
                        output["meta_info"]["input_token_logprobs"] = scores
                        # Retain exact candidate provenance without duplicating
                        # millions of candidate triples in every comparison.
                        top = output["meta_info"].pop("output_top_logprobs")
                        output["meta_info"]["candidate20_sha256"] = hashlib.sha256(
                            json.dumps(top).encode()
                        ).hexdigest()
                    outputs.extend(current)
                    cases.append(
                        {"concurrency": count, "input_lengths": list(map(len, probes[:count])), "output_tokens": 16384}
                    )
                    save(self.directories[index] / f"{label}-full-cap-progress.json", cases)
            paths = sorted(self.directories[index].glob("audit-*.json"), key=lambda p: p.stat().st_mtime_ns)
            if not paths:
                raise ValueError("Missing full-cap serving tensor audit")
            audit = json.loads(paths[-1].read_text())
            if getattr(self.args, "nvfp4_kv", False):
                pools = list(self.directories[index].glob("pool-audit-*.json"))
                if len(pools) != 1:
                    raise ValueError("Missing full-cap native KV audit")
                qualify_math_nvfp4_kv.verify_audit(json.loads(pools[0].read_text()))
                qualify_math_quantization.verify_inventory(
                    [m for m in audit if m["class"] != "RadixAttention"], self.args.precision
                )
            else:
                qualify_math_quantization.verify_inventory(audit, self.args.precision, kv_cache_dtype="bf16")
            return outputs, audit

        with futures.ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(receiver, range(3)))
        outputs, audits = map(list, zip(*results, strict=True))
        save(self.args.output / f"{label}.json", {"outputs": outputs, "audits": audits})
        return outputs, audits


def fresh_checkpoint(args, variant):
    if args.reuse_mutation_checkpoints is not None:
        target = args.reuse_mutation_checkpoints / f"changed-checkpoint-{variant}"
        receipt = json.loads((target / "qualification-mutation.json").read_text())
        if receipt["variant"] != variant or Path(receipt["source"]).resolve() != args.model.resolve():
            raise ValueError("Mutation checkpoint provenance mismatch")
        with (target / "model.safetensors").open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != receipt["sha256"]:
            raise ValueError("Mutation checkpoint hash mismatch")
        save(args.output / f"reused-mutation-{variant}.json", {"path": str(target), **receipt})
        return target
    target = args.output / f"changed-checkpoint-{variant}"
    target.mkdir()
    for source in args.model.iterdir():
        if source.is_file() and source.name != "model.safetensors":
            (target / source.name).symlink_to(source.resolve())
    # 25 GB host memory, released before starting a fresh server. Keep artifacts
    # until the study has verified completion; never overwrite the source model.
    with safe_open(args.model / "model.safetensors", framework="pt", device="cpu") as reader:
        names = reader.keys()
        state = {name: changed_tensor(name, reader.get_tensor(name), variant) for name in names}
        save_file(state, target / "model.safetensors", metadata={"format": "pt"})
    with (target / "model.safetensors").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    save(
        target / "qualification-mutation.json",
        {
            "variant": variant,
            "source": str(args.model),
            "sha256": digest,
            "scope": "routed experts only; qualification artifact, not study model",
        },
    )
    return target


def run(args):
    if os.environ.get("OI_PUBLICATION_NATIVE_CONFIG_RECEIPT"):
        raise ValueError("Native intervention is forbidden in strict refresh qualification")
    forbidden = [key for key, value in os.environ.items() if value and key.startswith("OI_PUBLICATION_TRACE_")]
    if forbidden:
        raise ValueError(f"Diagnostic tracing forbidden in strict refresh qualification: {forbidden}")
    args.output.mkdir(parents=True, exist_ok=False)
    configure_transport_environment()
    if torch.cuda.device_count() != 4:
        raise ValueError("Requires exactly the approved four visible GPUs")
    if any("B300" not in torch.cuda.get_device_name(i) for i in range(4)):
        raise ValueError("Requires the approved B300 hardware")
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port()}", rank=0, world_size=1)
    parallel = importlib.import_module("miles.backends.training_utils.parallel")
    group_type = importlib.import_module("miles.utils.ft_utils.process_group_utils").GroupInfo
    singleton = group_type(rank=0, size=1, group=None)
    parallel.set_parallel_state(
        parallel.ParallelState(
            **{key: singleton for key in ("intra_dp", "intra_dp_cp", "cp", "tp", "pp", "ep", "etp", "indep_dp")}
        )
    )
    publication = importlib.import_module("miles.backends.core_utils.publication.publication")
    async_utils = importlib.import_module("miles.utils.async_utils")
    tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    probes = (
        full_cap_probes(tokenizer, args.full_cap_prompts, args.full_cap_prompts_sha256)
        if getattr(args, "full_cap_prompts", None)
        else make_probes(tokenizer)
    )
    args.full_cap_conditional_inputs = {}
    with (args.model / "model.safetensors").open("rb") as stream:
        model_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    save(
        args.output / "protocol.json",
        {
            "transport": "miles.backends.core_utils.publication.publication.FlattenedDistributedUpdater",
            "source": str(args.model),
            "source_model_sha256": model_digest,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime_lock_sha256": hashlib.sha256(
                Path("/opt/core-rl/sources/runtime.lock.json").read_bytes()
            ).hexdigest(),
            "precision": args.precision,
            "transport_environment": {
                "NCCL_CUMEM_ENABLE": os.environ["NCCL_CUMEM_ENABLE"],
                "provenance": "Pinned miles.ray.specs.train.compute_trainer_env_vars sender default; inherited by servers",
            },
            "probes": probes,
            "full_cap_prompt_snapshot_sha256": getattr(args, "full_cap_prompts_sha256", None),
            "tolerances": TOLERANCES,
            "unchanged_bf16_tolerance": 0,
            "variants": [0, 1, 2, 2, 0],
            "scope": "Real Core NCCL transport and serving loader; HF checkpoint wire names. Core exporter and optimizer qualified separately.",
            "devices": [torch.cuda.get_device_name(i) for i in range(4)],
        },
    )
    changed = {v: fresh_checkpoint(args, v) for v in (1, 2)}
    observed, reports = {}, []
    with Fleet(args, args.model, "live") as fleet:
        baseline = fleet.probe("initial", probes)
        updater = publication.FlattenedDistributedUpdater(SimpleNamespace(rollout_num_gpus_per_engine=1))
        updater.connect_rollout_engines(fleet.clients, [1, 1, 1], [1, 2, 3])
        try:
            for version, variant in enumerate((0, 1, 2, 2, 0), start=1):
                print(json.dumps({"phase": "publication", "version": version, "variant": variant}), flush=True)
                for engine in fleet.clients:
                    async_utils.submit(engine.pause_generation("in_place")).result(timeout=120)
                    async_utils.submit(engine.begin_weight_update()).result(timeout=120)
                names, split_names, buckets = set(), set(), 0
                with safe_open(args.model / "model.safetensors", framework="pt", device="cpu") as reader:
                    for bucket in weight_buckets(reader, variant, 128 * 1024**2):
                        names.update(name for name, _ in bucket)
                        match = EXPERT.fullmatch(bucket[0][0])
                        if len(bucket) == 1 and match and match[2] == "0":
                            split_names.add(bucket[0][0])
                        updater.update_bucket_weights([(n, t.cuda().contiguous()) for n, t in bucket], version)
                        buckets += 1
                for engine in fleet.clients:
                    async_utils.submit(engine.end_weight_update()).result(timeout=600)
                    async_utils.submit(engine.update_weight_version(str(version))).result(timeout=120)
                    async_utils.submit(engine.continue_generation()).result(timeout=120)
                    actual_version = async_utils.submit(engine.get_weight_version()).result(timeout=120)
                    if int(actual_version) != version:
                        raise ValueError("Stale published version")
                current = fleet.probe(f"publication-{version}-variant-{variant}", probes)
                if sum(EXPERT.fullmatch(name) is not None for name in names) != 15 * 512 * 3:
                    raise ValueError("Incomplete per-expert publication name coverage")
                if len(split_names) != 15 * 3:
                    raise ValueError("Missing forced w1/w3 split-bucket coverage")
                if variant == 0:
                    for actual, expected in zip(current[0], baseline[0], strict=True):
                        compare_scores(actual, expected, args.precision, unchanged=args.precision in ("bf16", "fp8"))
                    if current[1] != baseline[1]:
                        raise ValueError("Unchanged/reverted publication changed serving weights")
                if variant in observed and current[1] != observed[variant][1]:
                    raise ValueError("Repeated publication changed scales/layout/weights")
                if variant in observed:
                    for actual, expected in zip(current[0], observed[variant][0], strict=True):
                        compare_scores(actual, expected, args.precision, unchanged=args.precision in ("bf16", "fp8"))
                observed[variant] = current
                report = {
                    "version": version,
                    "variant": variant,
                    "tensors": len(names),
                    "buckets": buckets,
                    "forced_split_names": sorted(split_names),
                }
                reports.append(report)
                save(args.output / "publication-progress.json", reports)
        finally:
            updater.close()
    for variant in (0, 1, 2):
        with Fleet(args, args.model if variant == 0 else changed[variant], f"fresh-{variant}") as fleet:
            reference = fleet.probe(f"fresh-{variant}", probes)
        actual = observed[variant]
        if actual[1] != reference[1]:
            raise ValueError(f"Fresh/updated parameter hashes differ for variant{variant}: scales/layout/weights")
        comparisons = [compare_scores(a, b, args.precision) for a, b in zip(actual[0], reference[0], strict=True)]
        # A changed-weight gate must actually perturb outputs on at least one
        # fixed token; equality to baseline cannot count as successful refresh.
        delta = max(
            abs(x[0] - y[0])
            for a, b in zip(actual[0], baseline[0], strict=True)
            for left, right in zip(a, b, strict=True)
            for x, y in zip(
                left["meta_info"]["input_token_logprobs"], right["meta_info"]["input_token_logprobs"], strict=True
            )
            if x[0] is not None and y[0] is not None
        )
        if variant != 0 and delta <= 1e-3:
            raise ValueError("Changed weights did not measurably perturb probe outputs")
        save(
            args.output / f"comparison-{variant}.json", {"comparisons": comparisons, "baseline_max_abs_change": delta}
        )
    save(
        args.output / "qualified.json",
        {
            "precision": args.precision,
            "publications": reports,
            "fresh_variants": [0, 1, 2],
            "dispatch_policy_sha256": hashlib.sha256(Path(os.environ["OI_MILES_KDA_POLICY"]).read_bytes()).hexdigest()
            if os.environ.get("OI_MILES_KDA_POLICY")
            else None,
            "passed": True,
            "nvfp4_kv": getattr(args, "nvfp4_kv", False),
            "full_cap_qualified": bool(getattr(args, "full_cap_prompts", None)),
            "scope": "Repeated actual Core publication and fresh-server parity; optimizer/checkpoint/resume remains a separate gate",
        },
    )
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "nvfp4", "fp8"), required=True)
    parser.add_argument("--nvfp4-kv", action="store_true")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--reuse-mutation-checkpoints", type=Path)
    parser.add_argument("--full-cap-prompts", type=Path)
    parser.add_argument("--full-cap-prompts-sha256")
    args = parser.parse_args()
    if bool(args.full_cap_prompts) != bool(args.full_cap_prompts_sha256):
        parser.error("Full-cap prompts require both snapshot path and sha256")
    if args.server:
        server(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
