# Copied from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/vllm_plugin/src/olmoe3_vllm_plugin/multimodal_weights.py
"""Tensor-name routing for whole-model (vision + OLMoE3 LM) exports served language-model-only.

A multimodal export (``model_type: olmo3moe_vision``) stores the language model as
``language_model.<text export name>`` and the olmo-core vision stack as ``vision_tower.*`` /
``connector.*``. vLLM only ever builds the language model from such an export, so this module
decides which tensors reach the text executor's loader and which are dropped. It imports
neither torch nor vLLM, so the contract is unit-testable in the CPU plugin environment.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Dict, Generic, Tuple, TypeVar

T = TypeVar("T")

LANGUAGE_MODEL_PREFIX = "language_model."
"""Prefix of the text executor's tensors in a multimodal export."""

TOWER_PREFIXES: Tuple[str, ...] = ("vision_tower.", "connector.")
"""Prefixes of the vision tensors, which are never loaded in language-model-only mode."""


class MultimodalWeightStream(Generic[T]):
    """
    Route a multimodal export's tensors for language-model-only serving.

    Iterating yields the language-model tensors with their ``language_model.`` prefix
    stripped, one at a time (the text executor's loader streams them; nothing is held back).
    ``vision_tower.*`` / ``connector.*`` tensors are dropped and counted in :attr:`skipped`.

    :param weights: ``(name, tensor)`` pairs as stored in the export.

    :raises KeyError: While iterating, on a tensor outside the ``language_model.`` /
        ``vision_tower.`` / ``connector.`` namespaces: the export does not follow the
        multimodal contract.
    """

    def __init__(self, weights: Iterable[Tuple[str, T]]):
        self._weights = weights
        self.language_model_tensors = 0
        """Number of ``language_model.*`` tensors yielded so far."""
        self.skipped: Dict[str, int] = {prefix: 0 for prefix in TOWER_PREFIXES}
        """Number of skipped tensors so far, keyed by tower prefix."""

    def __iter__(self) -> Iterator[Tuple[str, T]]:
        for name, tensor in self._weights:
            if name.startswith(LANGUAGE_MODEL_PREFIX):
                self.language_model_tensors += 1
                yield name[len(LANGUAGE_MODEL_PREFIX) :], tensor
                continue
            for prefix in TOWER_PREFIXES:
                if name.startswith(prefix):
                    self.skipped[prefix] += 1
                    break
            else:
                raise KeyError(
                    f"Unexpected Olmo3MoeVision checkpoint weight: {name!r} (expected a "
                    f"{LANGUAGE_MODEL_PREFIX!r} or {'/'.join(TOWER_PREFIXES)!r} tensor)"
                )

    def summary(self) -> str:
        """Human-readable routing counts, for the load log."""
        total = sum(self.skipped.values())
        parts = ", ".join(f"{count} {prefix}*" for prefix, count in self.skipped.items())
        return (
            f"{self.language_model_tensors} {LANGUAGE_MODEL_PREFIX}* tensor(s) read, "
            f"skipped {total} vision tensor(s) ({parts})"
        )


def check_language_model_loaded(loaded: Iterable[str], expected: Iterable[str]) -> set:
    """
    Assert the text executor received every parameter it owns.

    :param loaded: Parameter names (text executor namespace) its loader reported as loaded.
    :param expected: Every parameter name of the text executor.

    :returns: ``loaded`` re-prefixed with ``language_model.``, i.e. the names vLLM's strict
        loader compares with the wrapper's own parameters.

    :raises RuntimeError: Listing the parameters the export did not provide.
    """
    loaded_set = set(loaded)
    missing = sorted(set(expected) - loaded_set)
    if missing:
        preview = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise RuntimeError(
            f"{len(missing)} language-model parameter(s) were not initialized from the "
            f"multimodal export's {LANGUAGE_MODEL_PREFIX}* tensors: {preview}"
        )
    return {f"{LANGUAGE_MODEL_PREFIX}{name}" for name in loaded_set}


__all__ = [
    "LANGUAGE_MODEL_PREFIX",
    "TOWER_PREFIXES",
    "MultimodalWeightStream",
    "check_language_model_loaded",
]
