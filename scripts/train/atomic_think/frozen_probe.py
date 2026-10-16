"""Repeat H077 probes on frozen native checkpoints, without a trainer or optimizer."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from olmo_core import train
from torch import distributed as dist
from torch.distributed.checkpoint import FileSystemReader

from open_instruct import atomic_think, logger_utils, olmo_core_utils

logger = logger_utils.setup_logger(__name__)
ARMS = ("gate-reference", "gate-probed")
CONFIG = "scripts/train/debug/kda_hero_small_sft.json"
MANIFEST = "scripts/train/atomic_think/probes.json"


def normalize_hashes(hashes):
    result = {name.removeprefix("module."): value for name, value in hashes.items()}
    if len(result) != len(hashes):
        raise ValueError("Wrapper removal produced duplicate state keys")
    return result


def check_hashes(expected, actual):
    expected, actual = normalize_hashes(expected), normalize_hashes(actual)
    if expected.keys() != actual.keys():
        raise ValueError("Frozen model state keys differ from training")
    changed = [name for name in expected if expected[name] != actual[name]]
    if changed:
        raise ValueError(f"Frozen model differs from training: {len(changed)} tensors, first={changed[:5]}")


def state_hashes(model):
    return {name: atomic_think.tensor_hash(value) for name, value in model.state_dict().items()}


def check_checkpoint_coverage(module, checkpoint):
    metadata = FileSystemReader(checkpoint).read_metadata()
    keys = set(metadata.state_dict_metadata)
    missing = [
        name for name, _ in module.model.named_parameters() if module._resolve_model_checkpoint_key(name, keys) is None
    ]
    missing.extend(key for key in module._persistent_model_buffer_state_dict() if key not in keys)
    if missing:
        raise ValueError(f"Checkpoint omits model state: {missing[:5]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/output"))
    args = parser.parse_args()
    train.prepare_training_environment(seed=33333, shared_filesystem=True)
    try:
        if dist.get_world_size() != 1:
            raise ValueError("Frozen diagnostic requires exactly one process")
        config = olmo_core_utils.get_transformer_config(CONFIG, vocab_size=100352, attn_backend="flash_2")
        model = config.build(init_device="meta")
        module_config = olmo_core_utils.build_ddp_train_module_config(
            CONFIG,
            rank_microbatch_size=65536,
            max_sequence_length=65536,
            learning_rate=5e-5,
            weight_decay=0.0,
            scheduler=None,
            max_grad_norm=1.0,
            compile_model=False,
            ac_config=None,
        )
        module = module_config.build(model, eval_only=True)
        if module.optim is not None:
            raise RuntimeError("Frozen diagnostic unexpectedly built an optimizer")
        manifest_hash = hashlib.sha256(Path(MANIFEST).read_bytes()).hexdigest()
        report: dict[str, Any] = {
            "manifest_hash": manifest_hash,
            "world_size": 1,
            "arms": {},
            "scope": "Frozen inference only",
        }
        for arm in ARMS:
            checkpoint = args.root / arm / "step4" / "model_and_optim"
            expected = json.loads((args.root / arm / "atomic-audit" / "state-step4-rank0.json").read_text())
            check_checkpoint_coverage(module, str(checkpoint))
            start = time.monotonic()
            module.load_state_dict_direct(str(checkpoint), load_optim_state=False, thread_count=8)
            load_seconds = time.monotonic() - start
            before = state_hashes(module.model)
            check_hashes(expected["model"], before)
            logger.info("Frozen %s matches all %s training-state hashes", arm, len(before))
            arm_report = {
                "checkpoint": str(checkpoint),
                "load_seconds": load_seconds,
                "model_hashes": before,
                "repeats": [],
            }
            for repeat in range(3):
                output = args.output / arm / f"repeat{repeat}"
                callback = atomic_think.AtomicAuditCallback(
                    mode="control", horizon=11768, manifest=MANIFEST, steps=[4], output_dir=str(output)
                )
                callback.trainer = cast(Any, SimpleNamespace(train_module=module, global_step=4))
                torch.cuda.reset_peak_memory_stats()
                callback.pre_train()
                check_hashes(before, state_hashes(module.model))
                probe_report = json.loads((output / "probe-step4-rank0.json").read_text())
                arm_report["repeats"].append(
                    {**probe_report, "repeat": repeat, "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
                )
            report["arms"][arm] = arm_report
            atomic_think.write_json(args.output / "frozen-report.json", report)
    finally:
        train.teardown_training_environment()


if __name__ == "__main__":
    main()
