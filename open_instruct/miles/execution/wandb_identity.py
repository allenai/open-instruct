"""Keep one W&B run per output root across attempts.

MILES resumes a W&B run when ``wandb_run_id`` is set (``wandb.init(id=...,
resume="allow")``). The workflow owner resolves that id on the cluster, under the
run-directory lock, so preemption auto-resume and resubmission after a wall-clock
stop continue the same run instead of opening a new one.
"""

import contextlib
import hashlib
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

ID_FILE = "wandb_run_id"
FINGERPRINT_FILE = "wandb_run_id.json"


def config_hash(fields):
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()


def _write_atomic(path, text):
    # Temp file in the same directory, then rename: readers never see a partial id.
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def resolve(root, fields):
    """Return the W&B run id stored under ``root``, creating it on first start.

    ``fields`` are the recipe-relevant settings recorded with the id. A different
    hash on restart logs a warning; the stored id is still reused.
    """
    root = Path(root)
    path = root / ID_FILE
    digest = config_hash(fields)
    if path.is_file():
        run_id = path.read_text().strip()
        if run_id:
            stored = None
            # A missing or unreadable fingerprint only skips the mismatch check.
            with contextlib.suppress(OSError, ValueError, AttributeError):
                stored = json.loads((root / FINGERPRINT_FILE).read_text()).get("config_hash")
            if stored is not None and stored != digest:
                logger.warning(
                    "W&B RUN CONFIG MISMATCH: %s belongs to W&B run %s with a different model/data/W&B "
                    "configuration (stored hash %s, current %s). Continuing that run; use a new output.root "
                    "for a new run.",
                    root,
                    run_id,
                    stored[:12],
                    digest[:12],
                )
            logger.info("Resuming W&B run %s from %s", run_id, path)
            return run_id
    run_id = uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    _write_atomic(
        root / FINGERPRINT_FILE,
        json.dumps({"id": run_id, "created_at": time.time(), "config_hash": digest, "fields": fields}, default=str),
    )
    # The id file is written last: it is the commit point.
    _write_atomic(path, run_id + "\n")
    logger.info("Created W&B run id %s at %s", run_id, path)
    return run_id


def apply(root, spec, miles):
    """Set ``miles["wandb_run_id"]`` from ``root`` unless the user set one or W&B is off."""
    if not miles.get("use_wandb") or miles.get("wandb_run_id"):
        return miles.get("wandb_run_id")
    fields = {
        "model": spec.model.get("source"),
        "data": spec.data,
        "wandb_project": miles.get("wandb_project"),
        "wandb_team": miles.get("wandb_team"),
        "wandb_group": miles.get("wandb_group"),
    }
    miles["wandb_run_id"] = resolve(root, fields)
    return miles["wandb_run_id"]
