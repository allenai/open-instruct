"""Named training pools assigned by source or canonical top-level row labels."""

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


def resolve(data):
    """Infer source-assigned pools without changing the user's data configuration."""
    result = dict(data.get("pools", {})) if isinstance(data.get("pools", {}), dict) else {}
    if "pools" in data:
        validate(data["pools"])
    sources = data.get("sources", [])
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict) or "pool" not in source:
                continue
            name = source["pool"]
            if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
                raise InputError("Source pool must be a nonempty safe pool name")
            if name not in result:
                owner = next((pool for pool, spec in result.items() if spec["label"] == name), None)
                if owner is not None:
                    raise InputError(f"Source pool {name!r} collides with the label of configured pool {owner!r}")
                result[name] = {"label": name}
        if (
            result
            and "pools" not in data
            and any(isinstance(source, dict) and "pool" not in source for source in sources)
        ):
            raise InputError("Set pool on every source, or define data.pools for sources using row labels")
    if result:
        validate(result)
    return result


def annotate(row, pools, *, source_pool=None, labels=None):
    if labels is None:
        labels = {spec["label"]: name for name, spec in pools.items()}
    # Explicit source assignment wins over row labels in the prepared artifact.
    # This permits dataset-based experiments on the same topic-labeled inputs.
    label = pools[source_pool]["label"] if source_pool is not None else row.get("pool_id")
    if not isinstance(label, str) or label not in labels:
        raise InputError(f"Missing or unknown top-level pool_id: {label!r}")
    metadata = row["metadata"]
    if "pool_id" in metadata or "sampling_pool" in metadata:
        raise InputError("Pool runtime metadata is reserved; supply only top-level pool_id")
    row["pool_id"] = label
    # Dataset schema stays canonical; MILES Sample carries extra fields in metadata.
    metadata.update(pool_id=label, sampling_pool=labels[label])
