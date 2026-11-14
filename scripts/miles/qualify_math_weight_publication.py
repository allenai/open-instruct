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
from pathlib import Path
from types import SimpleNamespace

import httpx
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from scripts.miles import qualify_math_quantization
from torch import distributed as dist

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


def instrument_kda_inputs():
    """Retain first real prefill recurrence inputs; never change cache state."""
    if os.environ.get("OI_PUBLICATION_TRACE_KDA") != "1":
        return
    kernel = importlib.import_module("olmo_sglang.kda.backend").OlmoFLAKDAKernel
    original = kernel._run
    captured = False

    def run(instance, *values, **options):
        nonlocal captured
        if captured or not values or values[0].shape[1] < 32:
            return original(instance, *values, **options)
        captured = True
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
        torch.save(inputs, directory / f"kda-inputs-{os.getpid()}.pt")
        result = original(instance, *values, **options)
        output = result[0] if isinstance(result, tuple) else result
        torch.save(output.detach().cpu().clone(), directory / f"kda-output-{os.getpid()}.pt")
        save(
            directory / f"kda-receipt-{os.getpid()}.json",
            {"diagnostic_only": True, "synchronizes_cuda": True, "cache_modified_by_instrumentation": False},
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
    instrument_receiver()
    instrument_nvfp4_quantization()
    instrument_nvfp4_processing()
    instrument_kda_inputs()
    model_type = importlib.import_module("olmo_sglang.models.olmo3_moe").Olmo3MoeForCausalLM
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

        result = original_load(model, observed_weights(), *values[1:], **options)
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
        epoch = getattr(model, "_publication_audit_epoch", 0)
        if getattr(model, "_publication_audited_epoch", None) != epoch:
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
        return original_forward(model, *values, **options)

    model_type.load_weights, model_type.forward = load, forward
    return importlib.import_module("sglang.srt.managers.scheduler").run_scheduler_process(*args, **kwargs)


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
    if args.precision == "nvfp4":
        options.update(quantization="nvfp4_online", moe_runner_backend="flashinfer_trtllm_routed")
    return options


def server(args):
    os.environ["OLMO_SGLANG_CORE_COMPAT"] = "auto" if args.precision == "bf16" else "off"
    importlib.import_module("olmo_sglang").register()
    server_args = importlib.import_module("sglang.srt.server_args").ServerArgs
    options = server_options(args)
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
        result, audits = [], []
        with httpx.Client(timeout=600) as client:
            for engine, directory in zip(self.clients, self.directories, strict=True):
                response = client.post(
                    engine.server_url + "/generate",
                    json={
                        "input_ids": probes,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 1},
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
                qualify_math_quantization.verify_inventory(audit, self.args.precision)
                audits.append(audit)
        save(self.args.output / f"{label}.json", {"outputs": result, "audits": audits})
        return result, audits


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
    probes = make_probes(tokenizer)
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
                        compare_scores(actual, expected, args.precision, unchanged=args.precision == "bf16")
                    if current[1] != baseline[1]:
                        raise ValueError("Unchanged/reverted publication changed serving weights")
                if variant in observed and current[1] != observed[variant][1]:
                    raise ValueError("Repeated publication changed scales/layout/weights")
                if variant in observed:
                    for actual, expected in zip(current[0], observed[variant][0], strict=True):
                        compare_scores(actual, expected, args.precision, unchanged=args.precision == "bf16")
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
    for variant in (1, 2):
        with Fleet(args, changed[variant], f"fresh-{variant}") as fleet:
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
        if delta <= 1e-3:
            raise ValueError("Changed weights did not measurably perturb probe outputs")
        save(
            args.output / f"comparison-{variant}.json", {"comparisons": comparisons, "baseline_max_abs_change": delta}
        )
    save(
        args.output / "qualified.json",
        {"precision": args.precision, "publications": reports, "fresh_variants": [1, 2], "passed": True},
    )
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "nvfp4"), required=True)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--reuse-mutation-checkpoints", type=Path)
    args = parser.parse_args()
    if args.server:
        server(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
