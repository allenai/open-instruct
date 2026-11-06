"""Named training pools selected by the canonical top-level pool_id field."""

import re

from open_instruct.miles.errors import InputError


def validate(pools):
    if not isinstance(pools, dict) or not pools:
        raise InputError("data.pools must be a nonempty table of named labels")
    labels = set()
    for name, spec in pools.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise InputError("Pool names must be safe identifiers")
        if (
            not isinstance(spec, dict)
            or set(spec) != {"label"}
            or not isinstance(spec["label"], str)
            or not spec["label"]
        ):
            raise InputError(f"Pool {name} requires exactly one nonempty label")
        if spec["label"] in labels:
            raise InputError("Pool labels must be unique")
        labels.add(spec["label"])


def annotate(row, pools):
    labels = {spec["label"]: name for name, spec in pools.items()}
    label = row.get("pool_id")
    if not isinstance(label, str) or label not in labels:
        raise InputError(f"Missing or unknown top-level pool_id: {label!r}")
    metadata = row["metadata"]
    if "pool_id" in metadata or "sampling_pool" in metadata:
        raise InputError("Pool runtime metadata is reserved; supply only top-level pool_id")
    # Dataset schema stays canonical; MILES Sample carries extra fields in metadata.
    metadata.update(pool_id=label, sampling_pool=labels[label])
