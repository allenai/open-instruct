"""Opt-in H077 interventions and bounded, natural-route probes for native Core SFT."""

import contextlib
import functools
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from olmo_core.distributed.utils import get_rank, get_world_size, hide_from_torch, unhide_from_torch
from olmo_core.optim.moe_optimizer import assign_full_tensor_to_dtensor
from olmo_core.train.callbacks import Callback
from olmo_core.train.train_module.transformer.ddp_train_module import OLMoDDPTrainModule
from olmo_core.train.utils import EnvRngStates
from torch import nn
from torch.distributed.tensor import DTensor

from open_instruct import logger_utils, model_utils

logger = logger_utils.setup_logger(__name__)
MODES = ("control", "rounding-only", "atomic-safe", "atomic-historical")


def validate_config(sft, training, tokenizer_config):
    if sft.scheduler_horizon is not None:
        if training.lr_scheduler_type not in ("linear", "cosine"):
            raise ValueError("A fixed scheduler horizon requires linear or cosine scheduling")
        if sft.scheduler_horizon < (training.max_train_steps or 1):
            raise ValueError("scheduler_horizon must cover max_train_steps")
    if sft.atomic_audit_mode is None:
        return
    if sft.atomic_audit_mode not in MODES:
        raise ValueError(f"Unknown atomic audit mode: {sft.atomic_audit_mode}")
    atomic = sft.atomic_audit_mode.startswith("atomic-")
    if atomic != bool(tokenizer_config.reserved_slot_tokens):
        raise ValueError("Audit mode disagrees with reserved_slot_tokens")
    if atomic and tokenizer_config.reserved_slot_tokens != ["<think>", "</think>"]:
        raise ValueError("Atomic audit requires exactly the two thinking tags")
    if sft.scheduler_horizon is None:
        raise ValueError("Atomic audit requires an explicit scheduler_horizon")
    if training.compile_model:
        raise ValueError("Atomic audit uses the historical eager recipe")


def tensor_hash(value):
    value = value.to_local() if isinstance(value, DTensor) else value
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    return hashlib.sha256(raw).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def initialize_arm(train_module, tokenizer, mode, output_dir, sync_rows):
    """Apply one intervention, checking all local optimizer state outside target rows."""
    model, optimizer = train_module.model, train_module.optim
    if not hasattr(optimizer, "states"):
        raise ValueError("H077 requires native OLMoDDP FP32 optimizer masters")
    matrices = [model.embeddings.weight]
    if not model.tie_word_embeddings:
        matrices.append(model.lm_head.w_out.weight)
    targets = {id(p) for p in matrices}
    selected = {
        name: param
        for group in optimizer.param_groups
        for name, param in group["named_params"].items()
        if id(param) in targets
    }
    if len(selected) != len(targets):
        raise ValueError("Optimizer does not own every target matrix")
    rows = model_utils.promoted_token_rows(tokenizer)
    if bool(rows) != mode.startswith("atomic-"):
        raise ValueError("Tokenizer promotion metadata disagrees with audit mode")
    before_other = {
        name: tensor_hash(value)
        for name, value in optimizer.states.items()
        if name not in {f"{name}.main" for name in selected}
    }
    report: dict[str, Any] = {"mode": mode, "rank": get_rank(), "matrices": {}}
    for name, param in selected.items():
        master = optimizer.states[f"{name}.main"]
        original = master.full_tensor().reshape(param.shape)
        original = original.clone()
        before_model = param.detach().clone()
        if rows:
            model_utils.seed_embedding_rows(param, rows)
        if mode in ("rounding-only", "atomic-historical"):
            assign_full_tensor_to_dtensor(master, param.detach().float().reshape(-1))
        elif mode == "atomic-safe":
            sync_rows(optimizer, [param], [target for target, _ in rows])
        after = master.full_tensor().reshape(param.shape)
        keep = torch.ones(param.shape[0], dtype=torch.bool, device=original.device)
        if rows:
            keep[[target for target, _ in rows]] = False
        changed = (after[keep] != original[keep]).sum().item()
        if mode in ("control", "atomic-safe") and changed:
            raise RuntimeError("Unrelated master values changed in a precision-preserving arm")
        if not torch.equal(param.detach()[keep], before_model[keep]):
            raise RuntimeError("Unrelated model rows changed at initialization")
        if not torch.equal(after.to(param.dtype), param.detach()):
            raise RuntimeError("Master copy-back would alter the initialized model")
        report["matrices"][name] = {
            "unrelated_master_elements_changed": changed,
            "master_max_abs_change": (after - original).abs().max().item(),
            "model_hash_before": tensor_hash(before_model),
            "model_hash_after": tensor_hash(param),
            "master_hash_before": tensor_hash(original),
            "master_hash_after": tensor_hash(after),
        }
    after_other = {name: tensor_hash(optimizer.states[name]) for name in before_other}
    if before_other != after_other:
        raise RuntimeError("Intervention changed optimizer state outside the selected masters")
    report["other_optimizer_state_hashes"] = before_other
    write_json(Path(output_dir) / f"initialization-rank{get_rank()}.json", report)
    logger.info("H077 initialization verified: %s", mode)


@contextlib.contextmanager
def preserved_probe_state(model):
    """Restore RNG, training flags, buffers, and Core router's hidden mutable state."""
    rng = EnvRngStates.current_state()
    flags = [(m, m.training) for m in model.modules()]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    routers = [m for name, m in model.named_modules() if name.endswith("routed_experts_router")]
    hidden = []
    for router in routers:
        for key in ("_recompute_cache", "_debug_index"):
            value = getattr(router, key, None)
            if value is not None:
                raise ValueError(f"Probe would interrupt active router state: {key}")
        for key in (
            "batch_size_per_expert",
            "global_batch_size_per_expert",
            "score_bias_batch_size_per_expert",
            "load_balancing_loss",
            "z_loss",
            "orth_loss",
        ):
            key = "_" + key
            value = getattr(router, key)
            saved = None if value is None else hide_from_torch(unhide_from_torch(value).detach().clone())
            hidden.append((router, key, saved))
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
            for router, key, saved in hidden:
                setattr(router, key, saved)
            for router in routers:
                router._recompute_cache = None
                router._debug_index = None
        for module, training in flags:
            module.training = training
        if not rng.restore():
            raise RuntimeError("Could not restore RNG after routing probe")


@contextlib.contextmanager
def capture_routes(model):
    routes, statistics, handles, originals = {}, {}, [], []

    def observe(name, router, inputs, output):
        if name in routes:
            raise RuntimeError(f"Repeated router call during one probe: {name}")
        ids = output[1]
        if ids is None:
            raise RuntimeError("Expected selected experts, received scores-only routing")
        routes[name] = ids.detach().to(device="cpu", dtype=torch.int32)
        x = inputs[0].detach().float()
        statistics.setdefault(name, {})["input_rms"] = x.square().mean(-1).sqrt().cpu()
        statistics[name]["assignment_counts"] = torch.bincount(
            ids.flatten().long(), minlength=router.num_experts
        ).cpu()

    def top_k(original, name, router, scores):
        weights, ids = original(scores)
        # Grouped/alternative routing needs a separate margin definition; fail visibly
        # rather than reporting an unconstrained margin as the selection margin.
        if router.n_group is None and not router.uniform_expert_assignment and not router.random_expert_assignment:
            selection = scores.detach().clone()
            if router.score_bias is not None:
                selection += router.score_bias
            selected_min = selection.gather(-1, ids).min(-1).values
            selection.scatter_(-1, ids, -torch.inf)
            statistics.setdefault(name, {})["selection_margin"] = (selected_min - selection.max(-1).values).cpu()
        return weights, ids

    try:
        for name, router in model.named_modules():
            if name.endswith("routed_experts_router"):
                if getattr(router, "replay_expert_indices", None) is not None:
                    raise ValueError("Natural-route audit refuses forced route replay")
                handles.append(router.register_forward_hook(functools.partial(observe, name)))
                originals.append((router, "get_top_k" in router.__dict__, router.get_top_k))
                router.get_top_k = functools.partial(top_k, router.get_top_k, name, router)
        if not handles:
            raise ValueError("No routed-expert routers found")
        yield routes, statistics
    finally:
        for handle in handles:
            handle.remove()
        for router, owned, original in originals:
            if owned:
                router.get_top_k = original
            else:
                del router.get_top_k


@dataclass
class AtomicAuditCallback(Callback):
    mode: str = "control"
    horizon: int = 11768
    manifest: str | None = None
    steps: list[int] = field(default_factory=list)
    output_dir: str = "atomic-audit"
    state_steps: list[int] = field(default_factory=list)
    _manifest_hash: str = ""

    def state_dict(self):
        return {"mode": self.mode, "horizon": self.horizon, "manifest_hash": self._manifest_hash}

    def load_state_dict(self, state_dict):
        state = state_dict
        if state["mode"] != self.mode or state["horizon"] != self.horizon:
            raise ValueError("Resume changed the audit arm or scheduler horizon")
        self._manifest_hash = state["manifest_hash"]

    def pre_train(self):
        if self.manifest:
            current = hashlib.sha256(Path(self.manifest).read_bytes()).hexdigest()
            if self._manifest_hash and self._manifest_hash != current:
                raise ValueError("Resume changed the probe manifest")
            self._manifest_hash = current
        self.probe()

    def pre_step(self, batch):
        if self.step in self.steps or self.step <= 12 or self.step >= 510:
            hashes = {key: tensor_hash(value) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            write_json(Path(self.output_dir) / f"batch-step{self.step}-rank{get_rank()}.json", hashes)

    def post_step(self):
        self.probe()
        if self.step in self.state_steps:
            module = cast(OLMoDDPTrainModule, self.trainer.train_module)
            write_json(
                Path(self.output_dir) / f"state-step{self.step}-rank{get_rank()}.json",
                {
                    "model": {name: tensor_hash(p) for name, p in module.model.state_dict().items()},
                    "optimizer": {name: tensor_hash(p) for name, p in module.optim.states.items()},
                    "lr": [group["lr"] for group in module.optim.param_groups],
                    "step": self.step,
                },
            )

    def post_checkpoint_loaded(self, path):
        write_json(
            Path(self.output_dir) / f"restore-step{self.step}-rank{get_rank()}.json",
            {"checkpoint": str(path), "step": self.step, "mode": self.mode, "horizon": self.horizon},
        )

    def probe(self):
        if not self.manifest or self.step not in self.steps:
            return
        module = cast(OLMoDDPTrainModule, self.trainer.train_module)
        if any(getattr(module, name, False) for name in ("cp_enabled", "tp_enabled", "pp_enabled", "ep_enabled")):
            raise ValueError("H077 probe currently requires the historical data-parallel geometry")
        documents = json.loads(Path(self.manifest).read_text())["documents"]
        token_key = "atomic_ids" if self.mode.startswith("atomic-") else "original_ids"
        model = module.model
        device = next(model.parameters()).device
        results = []
        started = time.monotonic()
        with preserved_probe_state(model):
            for index in range(get_rank(), len(documents), get_world_size()):
                document = documents[index]
                ids = torch.tensor([document[token_key]], dtype=torch.long, device=device)
                with capture_routes(model) as (routes, statistics):
                    logits = model(ids)
                losses = nn.functional.cross_entropy(
                    logits[:, :-1].float().flatten(0, 1), ids[:, 1:].flatten(), reduction="none"
                )
                results.append(
                    {
                        "id": document["id"],
                        "kind": document["kind"],
                        "input_ids": ids.cpu(),
                        "doc_lens": torch.tensor([ids.shape[1]]),
                        "valid_mask": torch.ones(ids.shape, dtype=torch.bool),
                        "routes": {name: values.reshape(*ids.shape, -1) for name, values in routes.items()},
                        "statistics": statistics,
                        "token_nll": losses.cpu(),
                        "last_logits": logits[0, -1].detach().float().cpu(),
                    }
                )
                del logits, losses
        path = Path(self.output_dir) / f"probe-step{self.step}-rank{get_rank()}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"step": self.step, "manifest_hash": self._manifest_hash, "documents": results}, path)
        elapsed = time.monotonic() - started
        logger.info("H077 probe step %s rank %s: %s documents, %.2fs", self.step, get_rank(), len(results), elapsed)
        write_json(path.with_suffix(".json"), {"step": self.step, "seconds": elapsed, "documents": len(results)})
