"""Read-only provenance checks for a completed initial Megatron conversion.

File size/mtime and metadata/config hashes bind a previously audited asset; they
are not whole-model checksums or independent evidence of an HF Hub revision.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from open_instruct.miles import validation
from open_instruct.miles.errors import InputError

FIELDS = {
    "schema_version",
    "checkpoint_path",
    "hf_checkpoint_source",
    "original_model_revision",
    "hf_config_sha256",
    "architecture",
    "architecture_args_sha256",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "format",
    "tracker",
    "native_converter_sha256",
    "metadata_sha256",
    "checkpoint_files",
    "save_experiment",
    "load_experiment",
    "load_latest_job",
    "load_verified_utc",
    "all_model_parameters_finite",
    "sampled_hf_weights_equal",
    "training_updates",
    "scope",
}
REQUIRED_FILES = {
    "latest_checkpointed_iteration.txt",
    "release/metadata.json",
    "release/.metadata",
    "release/__0_0.distcp",
    "release/__1_0.distcp",
}


def _separate(checkpoint, output):
    for key in ("root", "assets"):
        target = Path(output[key])
        if checkpoint == target or checkpoint in target.parents or target in checkpoint.parents:
            raise InputError("Reused native checkpoint must be separate from output.root and output.assets")


def validate(provenance, model, trainer, output):
    """Check declarations without requiring mounted runtime files during plan/validate."""
    validation.mapping(provenance, "model.native_checkpoint")
    if not provenance:
        return
    validation.fields(provenance, "model.native_checkpoint", FIELDS)
    if set(provenance) != FIELDS:
        raise InputError("model.native_checkpoint requires the complete save/load provenance record")
    if type(provenance["schema_version"]) is not int or provenance["schema_version"] != 1:
        raise InputError("Native checkpoint provenance requires schema_version=1")
    for key in ("checkpoint_path", "hf_checkpoint_source"):
        value = validation.text(provenance[key], f"model.native_checkpoint.{key}")
        if not Path(value).is_absolute() or ".." in Path(value).parts:
            raise InputError(f"Native checkpoint {key} must be an absolute path without '..'")
    if Path(provenance["hf_checkpoint_source"]) != Path(model["source"]):
        raise InputError("Native checkpoint HF source differs from model.source")
    if provenance["architecture"] != model["architecture"]:
        raise InputError("Native checkpoint architecture differs from model.architecture")
    for key, expected in (("tensor_parallel_size", 2), ("pipeline_parallel_size", 1), ("training_updates", 0)):
        if type(provenance[key]) is not int or provenance[key] != expected:
            raise InputError("Initial asset reuse requires TP2/PP1 and zero prior training updates")
    if trainer["tensor_parallel_size"] != provenance["tensor_parallel_size"]:
        raise InputError("Native checkpoint tensor parallelism differs from trainer.tensor_parallel_size")
    if provenance["format"] != "torch_dist" or provenance["tracker"] != "release":
        raise InputError("Initial asset reuse requires a completed native torch_dist release")
    for key in ("hf_config_sha256", "native_converter_sha256", "metadata_sha256", "architecture_args_sha256"):
        if not isinstance(provenance[key], str) or not re.fullmatch(r"[0-9a-f]{64}", provenance[key]):
            raise InputError(f"Native checkpoint {key} must be a SHA256 hex digest")
    revision = provenance["original_model_revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise InputError("Native checkpoint original_model_revision must be a pinned revision")
    if not Path(provenance["hf_checkpoint_source"]).name.endswith(revision[:12]):
        raise InputError("Initial asset reuse requires the original HF source directory to retain its revision suffix")
    for key in ("save_experiment", "load_experiment", "load_latest_job"):
        if not isinstance(provenance[key], str) or not re.fullmatch(r"[0-9A-Z]{26}", provenance[key]):
            raise InputError(f"Native checkpoint {key} requires a retained Beaker identity")
    try:
        timestamp = datetime.fromisoformat(provenance["load_verified_utc"].replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        raise InputError("Native checkpoint load_verified_utc must be a UTC timestamp") from None
    if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise InputError("Native checkpoint load_verified_utc must be a UTC timestamp")
    for key in ("all_model_parameters_finite", "sampled_hf_weights_equal"):
        if provenance[key] is not True:
            raise InputError("Initial asset reuse requires retained finite-parameter and sampled-HF load audits")
    validation.text(provenance["scope"], "model.native_checkpoint.scope")
    files = validation.mapping(provenance["checkpoint_files"], "model.native_checkpoint.checkpoint_files")
    if set(files) != REQUIRED_FILES:
        raise InputError("TP2/PP1 initial release requires the exact five-file completed checkpoint inventory")
    for name, stat in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise InputError("Unsafe native checkpoint inventory path")
        if not isinstance(stat, list) or len(stat) != 2:
            raise InputError("Native checkpoint inventory values require [size_bytes, mtime_ns]")
        for value in stat:
            validation.integer(value, "native checkpoint file stat")
    _separate(Path(provenance["checkpoint_path"]), output)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checkpoint(provenance, output):
    """Reject partial or changed assets; never write the old asset or silently reconvert."""
    checkpoint = Path(provenance["checkpoint_path"])
    _separate(checkpoint.resolve(), {key: str(Path(value).resolve()) for key, value in output.items()})
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise InputError("Native checkpoint must be an existing non-symlink directory")
    actual = {}
    for path in checkpoint.rglob("*"):
        if path.is_symlink():
            raise InputError("Native checkpoint inventory contains a symlink")
        if path.is_file():
            stat = path.stat()
            actual[path.relative_to(checkpoint).as_posix()] = [stat.st_size, stat.st_mtime_ns]
    if actual != provenance["checkpoint_files"]:
        raise InputError("Native checkpoint file inventory changed or is incomplete")
    if (checkpoint / "latest_checkpointed_iteration.txt").read_text().strip() != "release":
        raise InputError("Native checkpoint completion tracker is not release")
    if _sha(checkpoint / "release/.metadata") != provenance["metadata_sha256"]:
        raise InputError("Native checkpoint metadata hash differs from the loaded asset")
    return checkpoint


def reuse(provenance, model, trainer, output, prepared_model, converter_sha, architecture_args):
    """Bind runtime input/config/converter identity to the independently load-audited release."""
    validate(provenance, model, trainer, output)
    if not provenance:
        raise InputError("Native checkpoint reuse requires explicit provenance")
    if Path(provenance["hf_checkpoint_source"]).resolve() != Path(model["source"]).resolve():
        raise InputError("Native checkpoint original HF source changed")
    if _sha(Path(prepared_model) / "config.json") != provenance["hf_config_sha256"]:
        raise InputError("Native checkpoint original HF config hash differs")
    if converter_sha != provenance["native_converter_sha256"]:
        raise InputError("Native checkpoint converter hash differs from the runtime image")
    architecture_sha = hashlib.sha256(json.dumps(architecture_args, sort_keys=True).encode()).hexdigest()
    if architecture_sha != provenance["architecture_args_sha256"]:
        raise InputError("Native checkpoint architecture arguments differ from the audited conversion")
    checkpoint = verify_checkpoint(provenance, output)
    return checkpoint, {
        "passed": True,
        "mode": "reuse",
        "checkpoint_path": str(checkpoint),
        "provenance": provenance,
        "note": "Read-only completed initial asset; no optimizer/cursor restore or whole-model parity claim.",
    }
