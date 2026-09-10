"""MILES custom reward hook for open-instruct's existing verifier classes.

``core.reward_config`` points to a trusted JSON mapping of verifier names to
``{"factory": "package.Class", "config": {...}}``. Samples carry only names,
weights, and targets in ``metadata.verifiers``; data cannot choose code to import.
"""

import asyncio
import functools
import importlib
import json
import math
from pathlib import Path


def _finite(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


@functools.lru_cache(maxsize=8)
def _registry(path):
    document = json.loads(Path(path).read_text())
    if not isinstance(document, dict) or not document:
        raise ValueError("reward_config must map verifier names to factory/config objects")
    result = {}
    for name, spec in document.items():
        module, _, symbol = spec["factory"].rpartition(".")
        factory = getattr(importlib.import_module(module), symbol)
        config = factory.get_config_class()(**spec.get("config", {}))
        result[name] = factory(verifier_config=config, **spec.get("kwargs", {}))
    return result


async def _score(args, sample):
    registry = _registry(args.olmo_core.reward_config)
    metadata = sample.metadata
    if not isinstance(metadata, dict) or not isinstance(metadata.get("verifiers"), list) or not metadata["verifiers"]:
        raise ValueError("Each sample requires nonempty metadata.verifiers")
    components = []
    total = 0.0
    for spec in metadata["verifiers"]:
        name = spec["name"]
        if name not in registry:
            raise ValueError(f"Verifier {name!r} is absent from the trusted reward registry")
        weight = _finite(spec.get("weight", 1.0), "weight")
        # response_length includes tool observations. Their tokens remain in the
        # trajectory, while the policy loss uses the separate MILES loss mask.
        tokens = sample.tokens[-sample.response_length :] if sample.response_length else []
        result = await registry[name].async_call(
            tokens,
            sample.response,
            spec["target"],
            query=metadata.get("query", sample.prompt),
            rollout_state=metadata.get("rollout_state"),
        )
        score = _finite(result.score, "verifier score")
        total += weight * score
        components.append({"name": name, "score": score, "weight": weight, "cost": result.cost})
    total = _finite(total, "combined reward")
    metadata["reward_components"] = components
    return total


async def registered_reward(args, samples, **kwargs):
    if not args.olmo_core.reward_config:
        raise ValueError("Set core.reward_config for the open-instruct verifier adapter")
    if isinstance(samples, list):
        return list(await asyncio.gather(*(_score(args, sample) for sample in samples)))
    return await _score(args, samples)
