"""Read-only probes of the actual Core/Megatron scorer, installed before Ray registration."""

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import torch
from torch import distributed as dist


def positions(length):
    return sorted(set(range(min(16, length))) | set(range(max(0, length - 128), length)))


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def prefix_payload(inputs):
    return {
        "schema_version": 1,
        "source": "frozen-serving-prefixes",
        "cases": [
            dict(
                case_id=x["case_id"],
                input_ids=list(x["input_ids"]),
                response_length=len(x["input_ids"]) - 1,
                loss_mask=[1] * (len(x["input_ids"]) - 1),
            )
            for x in inputs["cases"]
        ],
    }


def rollout_payload(samples):
    """Accept trusted, weights_only-loaded Sample.to_dict records without retokenization."""
    cases = []
    for index, sample in enumerate(samples):
        response = sample["response_length"]
        mask = sample.get("loss_mask")
        cases.append(
            dict(
                case_id=f"rollout0-sample{index}",
                input_ids=list(sample["tokens"]),
                response_length=response,
                loss_mask=list(mask) if mask is not None else [1] * response,
            )
        )
    return {"schema_version": 1, "source": "retained-rollout-0", "cases": cases}


def validate_payload(payload, world, max_length):
    if payload.get("schema_version") != 1 or not 1 <= len(payload.get("cases", [])) <= 16:
        raise ValueError("Expected schema1 and 1..16 explicit cases")
    cases = payload["cases"]
    if len(cases) % world or len({x["case_id"] for x in cases}) != len(cases):
        raise ValueError("Unique cases must divide evenly across trainer ranks")
    for case in cases:
        tokens, response, mask = case["input_ids"], case["response_length"], case["loss_mask"]
        if not tokens or len(tokens) > max_length or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("Invalid explicit token IDs or context overflow")
        if type(response) is not int or not 0 < response < len(tokens):
            raise ValueError("Response must have an explicit nonempty next-token prediction span")
        if len(mask) != response or any(type(x) not in (int, float) or x not in (0, 1) for x in mask):
            raise ValueError("Loss mask must preserve the response-token axis")
    return cases


def layer_id(name, backend):
    parts = name.split(".")
    anchor = "blocks" if backend == "olmo_core" else "layers"
    if anchor not in parts:
        raise ValueError(f"Cannot map native router to a canonical layer: {name}")
    return int(parts[parts.index(anchor) + 1])


def canonical_routes(logits, ids, weights, keep, total):
    logits = logits.detach().reshape(-1, logits.shape[-1])[:total]
    ids = ids.detach().reshape(-1, ids.shape[-1])[:total]
    weights = weights.detach().reshape(-1, weights.shape[-1])[:total]
    if logits.shape[0] != total or ids.shape != weights.shape:
        raise ValueError("Native routing output does not cover real tokens")
    if not torch.isfinite(logits).all() or not torch.isfinite(weights).all():
        raise ValueError("Nonfinite router observation")
    order = ids.argsort(-1)
    return {
        "logits": logits[keep].cpu().clone(),
        "topk_ids": ids.gather(-1, order)[keep].cpu().clone(),
        "topk_weights": weights.gather(-1, order)[keep].cpu().clone(),
    }


class Recorder:
    """Observe one real forward per case; preserve hooks and bound methods on exit."""

    def __init__(self, model, cases, backend):
        self.model, self.cases, self.backend = model, cases, backend
        self.records, self.index, self.handles, self.restores = [], -1, [], []
        self.logits = {}
        self.errors = []
        self.router_parameters = {}

    def begin(self, _module, args, kwargs):
        self.index += 1
        if self.index >= len(self.cases):
            raise ValueError("Unexpected extra scorer forward")
        case = self.cases[self.index]
        tokens = kwargs.get("input_ids", kwargs.get("input_", args[0] if args else None))
        if (
            not isinstance(tokens, torch.Tensor)
            or tokens.reshape(-1)[: len(case["input_ids"])].tolist() != case["input_ids"]
        ):
            raise ValueError("Actual model input differs from immutable case tokens")
        self.records.append(
            {
                "case_id": case["case_id"],
                "positions": positions(len(case["input_ids"])),
                "forward_token_count": tokens.numel(),
                "routes": {},
                "controls": {
                    "grad_enabled": torch.is_grad_enabled(),
                    "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                },
            }
        )

    def finish(self, _module, _args, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[0] != 1:
            raise ValueError("Expected TP1/PP1 batch-one vocabulary logits")
        case = self.cases[self.index]
        response = case["response_length"]
        selected = positions(response)
        target_positions = [len(case["input_ids"]) - response + i for i in selected]
        rows = torch.tensor([i - 1 for i in target_positions], device=output.device)
        targets = torch.tensor([case["input_ids"][i] for i in target_positions], device=output.device)
        logits = output[0, rows].float()
        selected_logits = logits.gather(-1, targets[:, None]).squeeze(-1)
        self.records[-1].update(
            response_positions=selected,
            target_token_positions=target_positions,
            target_logits=selected_logits.cpu(),
            log_normalizers=logits.logsumexp(-1).cpu(),
            reference_log_probs=(selected_logits - logits.logsumexp(-1)).cpu(),
            next_token_logits=output[0, len(case["input_ids"]) - 1].detach().cpu().clone(),
        )

    def route(self, name, output):
        case, record = self.cases[self.index], self.records[-1]
        if self.backend == "olmo_core":
            weights, ids, _, auxiliary = output
            logits = auxiliary[1]
        else:
            probabilities, routing_map = output
            logits = self.logits.pop(name)
            counts = routing_map.sum(-1)
            real = len(case["input_ids"])
            k = int(counts.reshape(-1)[0])
            if k <= 0 or not bool((counts.reshape(-1)[:real] == k).all()):
                raise ValueError("Expected fixed top-k routing on every real token")
            # Ignore artificial padding before constructing fixed-width expert IDs.
            routing_map = routing_map.reshape(-1, routing_map.shape[-1])[:real]
            ids = routing_map.nonzero()[:, -1].reshape(real, k)
            probabilities = probabilities.reshape(-1, probabilities.shape[-1])[:real]
            weights = probabilities.gather(-1, ids)
        layer = layer_id(name, self.backend)
        if layer in record["routes"]:
            raise ValueError("Duplicate native router invocation in one scorer forward")
        record["routes"][layer] = canonical_routes(logits, ids, weights, record["positions"], len(case["input_ids"]))

    def guarded(self, operation):
        def observe(*args, **kwargs):
            try:
                operation(*args, **kwargs)
            except Exception as exc:
                # Observation must never abort one rank before another enters EP communication.
                self.errors.append(f"{type(exc).__name__}: {exc}")
            return None

        return observe

    def __enter__(self):
        self.handles.append(self.model.register_forward_pre_hook(self.guarded(self.begin), with_kwargs=True))
        self.handles.append(self.model.register_forward_hook(self.guarded(self.finish)))
        for name, module in self.model.named_modules():
            wanted = (
                name.endswith(".routed_experts_router")
                if self.backend == "olmo_core"
                else name.endswith(".mlp.router")
            )
            if not wanted:
                continue
            weight = module.weight.detach()
            if hasattr(weight, "to_local"):
                weight = weight.to_local()
            self.router_parameters[layer_id(name, self.backend)] = {
                "native_name": name,
                "shape": list(weight.shape),
                "dtype": str(weight.dtype),
                "sha256": hashlib.sha256(weight.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
            }
            if self.backend == "megatron":
                original = module.gating

                def gating(*args, original=original, name=name, **kwargs):
                    value = original(*args, **kwargs)
                    self.logits[name] = value
                    return value

                module.gating = gating
                self.restores.append((module, original))
            self.handles.append(
                module.register_forward_hook(self.guarded(lambda _m, _a, out, name=name: self.route(name, out)))
            )
        if len(self.handles) == 2:
            self.__exit__(None, None, None)
            raise ValueError("No routed native modules found")
        return self

    def validate(self):
        if self.errors:
            raise ValueError("Observer failed: " + "; ".join(self.errors))
        expected = set(self.router_parameters)
        if any(set(record["routes"]) != expected for record in self.records):
            raise ValueError("Scorer observation omitted an expected routed layer")

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        for module, original in self.restores:
            module.gating = original


def assert_scores(records, cases, control, observed):
    if len(records) != len(cases) or len(control) != len(cases) or len(observed) != len(cases):
        raise ValueError("Scorer did not cover every immutable case")
    for record, case, before, after in zip(records, cases, control, observed, strict=True):
        before, after = before.detach().cpu(), after.detach().cpu()
        if before.shape != (case["response_length"],) or after.shape != before.shape:
            raise ValueError("Production log-probability response axis changed")
        if not torch.equal(before, after):
            raise ValueError("Scorer observations changed the production log probabilities")
        reference = record["reference_log_probs"]
        difference = (after[record["response_positions"]].float() - reference).abs()
        if not bool(torch.isfinite(after).all()) or float(difference.max()) > 1e-4:
            raise ValueError("Production score fails explicit logits[t-1,target_t] check")
        record.update(
            log_probs=after,
            loss_mask=list(case["loss_mask"]),
            response_length=case["response_length"],
            input_ids_sha256=digest(case["input_ids"]),
            alignment_max_abs=float(difference.max()),
        )


def runtime_metadata():
    """Read loaded source identities and populated tuner choices; never invoke a kernel."""
    module_name = (
        "update_zero_capture" if "update_zero_capture" in sys.modules else "scripts.miles.update_zero_capture"
    )
    capture = importlib.import_module(module_name)
    sources = capture.source_manifest()
    for name, module in list(sys.modules.items()):
        if name.startswith(("olmo_core.", "megatron.", "miles.backends.", "open_instruct.miles.")):
            filename = getattr(module, "__file__", None)
            if filename and Path(filename).is_file():
                sources[name] = {"file": filename, "sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest()}
    return {
        "autotune_configs": capture.snapshot_autotune_configs(),
        "autotune_policy": capture.autotune_policy(),
        "sources": sources,
    }


def diagnostic_score_probe(self, payload, output):
    """Ray-callable method added only by the diagnostic driver before actor registration."""
    backend = self.args.train_backend
    rank, world = dist.get_rank(), dist.get_world_size()
    error, cases = None, None
    path = Path(output) / f"trainer-{backend}-rank{rank}.pt"
    try:
        if (
            backend not in ("olmo_core", "megatron")
            or self.args.use_rollout_routing_replay
            or self.args.use_routing_replay
        ):
            raise ValueError("Probe requires an actual replay-disabled Core or Megatron scorer")
        if any(
            getattr(self.args, name, 1) != 1
            for name in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size")
        ):
            raise ValueError("Probe supports TP1/PP1/CP1 only")
        if self.args.rollout_temperature != 1.0:
            raise ValueError("Raw-logit alignment probe requires the original temperature1 scoring recipe")
        cases = validate_payload(payload, world, self.args.rollout_max_context_len)[rank::world]
        if path.exists():
            raise ValueError("Use a new output directory; existing evidence must remain immutable")
    except Exception as exc:
        error = f"rank{rank}: {type(exc).__name__}: {exc}"
    errors = [None] * world
    dist.all_gather_object(errors, error)
    if any(errors):
        raise ValueError("; ".join(x for x in errors if x))
    tokens = [torch.tensor(x["input_ids"], dtype=torch.long, device="cuda") for x in cases]
    rollout = dict(
        tokens=tokens,
        total_lengths=[len(x) for x in tokens],
        response_lengths=[x["response_length"] for x in cases],
        loss_masks=[torch.tensor(x["loss_mask"], device="cuda") for x in cases],
    )
    if backend == "olmo_core":
        data = importlib.import_module("open_instruct.miles.data")
        batches = data.sample_batches(rollout, self.args.rollout_max_context_len)
        model = self.model

        def score():
            return self._score(self.train_module, batches, use_replay=False)
    else:
        data = importlib.import_module("miles.backends.training_utils.data")
        if len(self.model) != 1 or self.args.micro_batch_size != 1 or self.args.data_pad_size_multiplier != 1:
            raise ValueError("Megatron probe requires original BSHD microbatch1/pad-multiple1 recipe")
        rollout["max_seq_lens"] = [max(rollout["total_lengths"])] * len(cases)
        model = self.model[0]

        def score():
            iterator = data.DataIterator(rollout, micro_batch_size=1)
            return self.compute_log_prob([iterator], [len(cases)], rollout_id=0)["log_probs"]

    control = score()
    with Recorder(model, cases, backend) as recorder:
        observed = score()
    error = None
    try:
        recorder.validate()
        assert_scores(recorder.records, cases, control, observed)
    except Exception as exc:
        error = f"rank{rank}: {type(exc).__name__}: {exc}"
    dist.all_gather_object(errors, error)
    if any(errors):
        raise ValueError("; ".join(x for x in errors if x))
    record = dict(
        schema_version=1,
        backend=backend,
        rank=rank,
        world_size=world,
        optimizer_calls=0,
        probe_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_actor_source_sha256=hashlib.sha256(
            Path(importlib.import_module(type(self).__module__).__file__).read_bytes()
        ).hexdigest(),
        case_indices=list(range(rank, len(payload["cases"]), world)),
        partition="explicit rank-strided diagnostic cohort",
        payload_sha256=digest(payload),
        source=payload["source"],
        cases=recorder.records,
        router_parameters=recorder.router_parameters,
        runtime=runtime_metadata(),
        scope="Actual production scorer; no replay, no backward, no optimizer call; canonical router observations",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, path)
    summary = dict(
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cases=len(cases),
        optimizer_calls=0,
        payload_sha256=record["payload_sha256"],
    )
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def install(backend):
    module_name, class_name = (
        ("open_instruct.miles.actor", "OLMoCoreTrainRayActor")
        if backend == "olmo_core"
        else ("miles.backends.megatron_utils.actor", "MegatronTrainRayActor")
    )
    if backend not in ("olmo_core", "megatron"):
        raise ValueError("Unknown trainer backend")
    implementation = getattr(importlib.import_module(module_name), class_name)
    implementation.diagnostic_score_probe = diagnostic_score_probe


def worker_setup():
    """Chain the original Ray worker hook, then register the diagnostic method locally."""
    original = os.environ.get("OI_TRAINER_ROUTE_ORIGINAL_WORKER_HOOK")
    if original:
        module, name = original.rsplit(".", 1)
        getattr(importlib.import_module(module), name)()
    install(os.environ["OI_TRAINER_ROUTE_BACKEND"])
