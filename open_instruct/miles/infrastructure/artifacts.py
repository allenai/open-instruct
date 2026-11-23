"""Publish JSON receipts and state files without exposing partially written output.
Writers serialize into a unique temporary file beside the destination and replace
it atomically, cleaning up the temporary file on failure. Launch, preparation and
evaluation can share this helper without depending on the training runtime.
"""

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
