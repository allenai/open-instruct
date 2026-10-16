"""Measure a scientific H077 arm from saved native checkpoints, outside training."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from olmo_core import train
from scripts.train.atomic_think import fixed_kernels, frozen_probe, run
from torch import distributed as dist
from transformers import AutoTokenizer

from open_instruct import atomic_think, logger_utils

logger = logger_utils.setup_logger(__name__)
STEPS = (0, 1, 2, 10, 100, 353, 512)


def verify_tokenizer(tokenizer, documents, mode):
    key = "atomic_ids" if mode.startswith("atomic-") else "original_ids"
    for document in documents:
        if tokenizer.encode(document["text"], add_special_tokens=False) != document[key]:
            raise ValueError(f"Saved tokenizer differs from frozen inputs: {document['id']}")
    if mode.startswith("atomic-"):
        for tag, token_id in (("<think>", 100256), ("</think>", 100270)):
            if tokenizer.encode(tag, add_special_tokens=False) != [token_id]:
                raise ValueError(f"Wrong atomic tag ID: {tag}")
            if tokenizer.decode([token_id], skip_special_tokens=True) != tag:
                raise ValueError(f"Atomic tag disappears during decoding: {tag}")


def expected_state(root, arm, mode, step, manifest_hash):
    reports = []
    for rank in range(16):
        path = root / arm / "atomic-audit" / f"model-state-step{step}-rank{rank}.json"
        report = json.loads(path.read_text())
        if report["step"] != step or report["mode"] != mode or report["manifest_hash"] != manifest_hash:
            raise ValueError(f"Wrong checkpoint audit provenance: {path}")
        reports.append(frozen_probe.normalize_hashes(report["model"]))
    if any(report != reports[0] for report in reports[1:]):
        raise ValueError("Training model replicas disagree")
    return reports[0]


def verify_reference(module, kernels, output):
    """Require the calibration checkpoint to reproduce every saved document exactly."""
    reference = kernels.reference
    for name, field in ((frozen_probe.MANIFEST, "manifest_hash"), (frozen_probe.CONFIG, "model_config_sha256")):
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != reference[field]:
            raise ValueError(f"Kernel calibration input changed: {name}")
    checkpoint = reference["checkpoint"]
    frozen_probe.check_checkpoint_coverage(module, checkpoint)
    module.load_state_dict_direct(checkpoint, load_optim_state=False, thread_count=8)
    frozen_probe.check_hashes(reference["model_hashes"], frozen_probe.state_hashes(module.model))
    callback = atomic_think.AtomicAuditCallback(
        mode="control", horizon=11768, manifest=frozen_probe.MANIFEST, steps=[4], output_dir=str(output)
    )
    callback.trainer = cast(Any, SimpleNamespace(train_module=module, global_step=4))
    callback.pre_train()
    frozen_probe.check_hashes(reference["model_hashes"], frozen_probe.state_hashes(module.model))
    path = output / "probe-step4-rank0.pt"
    capture = torch.load(path, map_location="cpu", weights_only=True)
    signatures = fixed_kernels.capture_signatures(capture)
    if signatures != reference["reference_documents"]:
        changed = [key for key in signatures if signatures[key] != reference["reference_documents"].get(key)]
        raise ValueError(f"Calibration checkpoint did not reproduce exactly: {len(changed)} changed documents")
    if kernels.observed != set(kernels.choices):
        raise ValueError("Calibration did not exercise every recorded kernel key")
    with path.open("rb") as stream:
        capture_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = {
        "kernel_replay": kernels.receipt(),
        "checkpoint": checkpoint,
        "documents": signatures,
        "capture_sha256": capture_hash,
        "job_id": os.environ.get("BEAKER_JOB_ID"),
        "hostname": os.environ.get("BEAKER_NODE_HOSTNAME"),
    }
    atomic_think.write_json(output / "verification.json", receipt)
    logger.info("Calibration checkpoint reproduced all %s documents exactly", len(signatures))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("arm", choices=run.ARMS)
    parser.add_argument("--steps", type=int, nargs="+", default=STEPS)
    parser.add_argument("--output", type=Path, default=Path("/output"))
    args = parser.parse_args()
    if len(set(args.steps)) != len(args.steps) or any(step not in STEPS for step in args.steps):
        parser.error("Steps must be unique registered measurement steps")
    mode = "control" if args.arm == "control-repeat" else args.arm
    manifest_path = Path(frozen_probe.MANIFEST)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    documents = json.loads(manifest_path.read_text())["documents"]
    tokenizer = AutoTokenizer.from_pretrained(args.root / args.arm / "tokenizer", local_files_only=True)
    verify_tokenizer(tokenizer, documents, mode)
    train.prepare_training_environment(seed=33333, shared_filesystem=True)
    try:
        if dist.get_world_size() != 1:
            raise ValueError("Offline diagnostic requires exactly one process")
        module = frozen_probe.build_module()
        output = args.output / args.arm / "atomic-audit"
        kernels = fixed_kernels.FixedKernels()
        with kernels:
            reference_verification = verify_reference(module, kernels, output / "kernel-reference")
            measure_checkpoints(
                args, module, output, mode, manifest_path, manifest_hash, kernels, reference_verification
            )
    finally:
        train.teardown_training_environment()


def measure_checkpoints(args, module, output, mode, manifest_path, manifest_hash, kernels, reference_verification):
    for step in args.steps:
        expected = expected_state(args.root, args.arm, mode, step, manifest_hash)
        checkpoint = args.root / args.arm / f"step{step}" / "model_and_optim"
        frozen_probe.check_checkpoint_coverage(module, str(checkpoint))
        logger.info("Loading offline %s step %s from %s", args.arm, step, checkpoint)
        started = time.monotonic()
        module.load_state_dict_direct(str(checkpoint), load_optim_state=False, thread_count=8)
        load_seconds = time.monotonic() - started
        frozen_probe.check_hashes(expected, frozen_probe.state_hashes(module.model))
        callback = atomic_think.AtomicAuditCallback(
            mode=mode, horizon=11768, manifest=str(manifest_path), steps=[step], output_dir=str(output)
        )
        callback.trainer = cast(Any, SimpleNamespace(train_module=module, global_step=step))
        torch.cuda.reset_peak_memory_stats()
        callback.pre_train()
        frozen_probe.check_hashes(expected, frozen_probe.state_hashes(module.model))
        path = output / f"probe-step{step}-rank0.pt"
        with path.open("rb") as stream:
            capture_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        atomic_think.write_json(
            output / f"offline-step{step}.json",
            {
                "arm": args.arm,
                "mode": mode,
                "step": step,
                "manifest_hash": manifest_hash,
                "checkpoint": str(checkpoint),
                "model_hashes": expected,
                "training_world_size": 16,
                "capture_world_size": 1,
                "capture_sha256": capture_hash,
                "load_seconds": load_seconds,
                "probe": json.loads(path.with_suffix(".json").read_text()),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "kernel_replay": kernels.receipt(),
                "reference_verification": reference_verification,
            },
        )
        logger.info("Verified offline %s step %s: %s model tensors", args.arm, step, len(expected))


if __name__ == "__main__":
    main()
