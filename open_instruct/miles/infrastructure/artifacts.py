"""Atomic application receipts, independent of the training runtime."""

import json
import os
import uuid
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    """Publish only after the complete JSON has reached the filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    stream = temporary.open("x")
    try:
        with stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
