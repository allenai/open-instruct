"""Precision-preserving warmup-to-SFT initialization, distinct from SFT resume."""

import hashlib
import json
from pathlib import Path

import torch
from olmo_core.distributed.utils import get_rank
from olmo_core.optim.moe_optimizer import assign_full_tensor_to_dtensor
from olmo_core.train.callbacks import Callback
from safetensors import torch as safe_torch
from torch.distributed.tensor import DTensor

from open_instruct import atomic_think, logger_utils, model_utils

logger = logger_utils.setup_logger(__name__)
MATRICES = ("embeddings.weight", "lm_head.w_out.weight")


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def vocab_digest(tokenizer):
    return hashlib.sha256(
        json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_manifest(path, expected_sha, base, tokenizer, mode):
    if file_digest(path) != expected_sha:
        raise ValueError("Warmup manifest hash mismatch")
    manifest = json.loads(Path(path).read_text())
    if manifest["version"] != 1 or manifest["optimizer_policy"] != "fresh-sft":
        raise ValueError("Unsupported warmup format or optimizer handoff")
    if Path(manifest["base"]).resolve() != Path(base).resolve():
        raise ValueError("Warmup base differs from SFT base")
    if manifest["tokenizer_vocab_sha256"] != vocab_digest(tokenizer):
        raise ValueError("Warmup tokenizer differs from SFT tokenizer")
    if mode not in ("control", "atomic-safe") or manifest["encoding"] != (
        "atomic" if mode == "atomic-safe" else "original"
    ):
        raise ValueError("Warmup encoding disagrees with SFT mode")
    promoted = [target for target, _ in model_utils.promoted_token_rows(tokenizer)]
    if promoted != ([100256, 100270] if mode == "atomic-safe" else []):
        raise ValueError("Unexpected promoted token IDs")
    payload = Path(path).parent / "matrices.safetensors"
    if manifest["file"] != payload.name or file_digest(payload) != manifest["file_sha256"]:
        raise ValueError("Warmup matrix file hash mismatch")
    values = safe_torch.load_file(str(payload))
    if set(values) != set(MATRICES) or set(manifest["matrices"]) != set(MATRICES):
        raise ValueError("Warmup must contain exactly input and output matrices")
    for name, value in values.items():
        entry = manifest["matrices"][name]
        shape = (100352, 1024) if entry["scope"] == "full" else (2, 1024)
        if entry["scope"] not in ("rows", "full") or tuple(value.shape) != shape:
            raise ValueError("Wrong warmup matrix scope or shape")
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("Warmup matrices must be finite FP32")
        if entry["scope"] == "rows" and (manifest["encoding"] != "atomic" or entry["rows"] != promoted):
            raise ValueError("Row-only warmup must cover exactly the promoted think rows")
        if atomic_think.tensor_hash(value) != entry["tensor_sha256"]:
            raise ValueError("Warmup matrix tensor hash mismatch")
    return manifest, values


def inject(train_module, manifest, values, output_dir, manifest_sha):
    """Write FP32 masters directly, retaining fresh moments and every other state."""
    model, optimizer = train_module.model, train_module.optim
    if model.tie_word_embeddings or not hasattr(optimizer, "states"):
        raise ValueError("Warmup requires untied native OLMoDDP matrices and masters")
    owners = {"embeddings.weight": model.embeddings.weight, "lm_head.w_out.weight": model.lm_head.w_out.weight}
    named = {id(p): name for group in optimizer.param_groups for name, p in group["named_params"].items()}
    if any(id(p) not in named for p in owners.values()):
        raise ValueError("Optimizer does not own each warmed matrix")
    selected = {f"{named[id(p)]}.main" for p in owners.values()}
    # A fresh SFT stage does not inherit warmup or pretraining optimizer history.
    for name, value in optimizer.states.items():
        local = value.to_local() if isinstance(value, DTensor) else value
        if name.endswith((".step", ".exp_avg", ".exp_avg_sq")) and torch.count_nonzero(local).item():
            raise ValueError("Warmup injection requires fresh SFT optimizer state")
    before = {
        name: atomic_think.tensor_hash(value) for name, value in optimizer.states.items() if name not in selected
    }
    report = {"manifest_sha256": manifest_sha, "optimizer_policy": "fresh-sft", "matrices": {}}
    for name, param in owners.items():
        entry = manifest["matrices"][name]
        master = optimizer.states[f"{named[id(param)]}.main"]
        original = master.full_tensor().reshape(param.shape)
        if original.dtype != torch.float32 or atomic_think.tensor_hash(original) != entry["base_master_sha256"]:
            raise ValueError("Base FP32 master differs from the qualified warmup source")
        if not torch.equal(original.to(param.dtype), param.detach()):
            raise ValueError("Base model disagrees with its FP32 master")
        if entry["scope"] == "rows":
            expected = original.clone()
            expected[entry["rows"]] = values[name].to(expected.device)
        else:
            expected = values[name].to(device=original.device)
        if expected.shape != original.shape or expected.dtype != torch.float32:
            raise ValueError("Warmup handoff shape/dtype mismatch")
        assign_full_tensor_to_dtensor(master, expected.reshape(-1))
        with torch.no_grad():
            param.copy_(expected.to(param.dtype))
        restored = master.full_tensor().reshape(param.shape)
        if not torch.equal(restored, expected) or not torch.equal(param.detach(), expected.to(param.dtype)):
            raise ValueError("Warmup handoff altered FP32 values or BF16 forward values")
        report["matrices"][name] = {
            "scope": entry["scope"],
            "master_sha256": atomic_think.tensor_hash(restored),
            "model_sha256": atomic_think.tensor_hash(param),
        }
    after = {name: atomic_think.tensor_hash(optimizer.states[name]) for name in before}
    if before != after:
        raise ValueError("Warmup injection changed unrelated optimizer state")
    report["other_optimizer_state_hashes"] = before
    atomic_think.write_json(Path(output_dir) / f"warmup-handoff-rank{get_rank()}.json", report)
    logger.info("Loaded warmed FP32 input/head matrices; retained fresh SFT optimizer state")


class WarmupIdentity(Callback):
    """Pin initialization identity in every SFT checkpoint; never reinject on resume."""

    def __init__(self, manifest_sha):
        self.manifest_sha = manifest_sha
        self.restored = False

    def state_dict(self):
        return {"manifest_sha256": self.manifest_sha, "optimizer_policy": "fresh-sft"}

    def load_state_dict(self, state_dict):
        if state_dict != self.state_dict():
            raise ValueError("Resume changed warmup initialization identity")
        self.restored = True

    def pre_train(self):
        if self.trainer.global_step > 0 and not self.restored:
            raise ValueError("SFT resume is missing its warmup identity")
