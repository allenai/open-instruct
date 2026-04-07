"""Registration for custom verifiers and verifier configs.

``@register_verifier`` and ``@register_verifier_config`` record classes merged by
:func:`open_instruct.ground_truth_utils.build_all_verifiers` and
:func:`build_streaming_config_with_registered_verifier_fields` (with subclass discovery for verifiers).

These APIs are re-exported from :mod:`open_instruct.ground_truth_utils` for convenience.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from open_instruct.ground_truth_utils import VerifierConfig, VerifierFunction

TVerifier = TypeVar("TVerifier", bound="VerifierFunction")
TVCfg = TypeVar("TVCfg", bound="VerifierConfig")

# Classes listed here are included in ``build_all_verifiers`` (merged with subclass discovery; deduped).
_decorated_verifier_classes: list[type[VerifierFunction]] = []

_registered_verifier_config_classes: list[type[VerifierConfig]] = []


def register_verifier(cls: type[TVerifier]) -> type[TVerifier]:
    """Opt in a concrete ``VerifierFunction`` subclass for ``build_all_verifiers``.

    Core verifiers in ``ground_truth_utils`` are still found via ``VerifierFunction.__subclasses__()``
    once imported. Use this decorator on **custom** verifiers (e.g. under ``examples/``) to make
    registration explicit and discoverable in code review.

    The defining module must still be imported before ``build_all_verifiers`` runs (e.g. import
    your ``verifier`` module from a launch script or ``examples/grpo_fast.py``).

    Example::

        @register_verifier
        class MyTaskVerifier(VerifierFunction):
            ...
    """
    _decorated_verifier_classes.append(cls)
    return cls


def register_verifier_config(cls: type[TVCfg]) -> type[TVCfg]:
    """Register a ``VerifierConfig`` subclass so its fields can be merged into streaming config.

    Use with :func:`build_streaming_config_with_registered_verifier_fields` in a launch script that
    subclasses ``StreamingDataLoaderConfig`` with the same field names as your verifier config
    (so ``build_all_verifiers`` can read them from ``streaming_config``).

    Apply **below** ``@dataclass`` so the decorated class is fully built::

        @register_verifier_config
        @dataclass
        class MyVerifierConfig(VerifierConfig):
            my_api_url: str = ...
    """
    _registered_verifier_config_classes.append(cls)
    return cls


def get_registered_verifier_config_classes() -> tuple[type[VerifierConfig], ...]:
    """Verifier config classes registered with :func:`register_verifier_config`."""
    return tuple(_registered_verifier_config_classes)


def build_streaming_config_with_registered_verifier_fields(*, class_name: str) -> type:
    """Subclass ``StreamingDataLoaderConfig`` with fields from all registered ``VerifierConfig`` classes."""
    # Defer heavy ``data_loader`` import; only needed when building the merged dataclass.
    from open_instruct.data_loader import StreamingDataLoaderConfig  # noqa: PLC0415
    from open_instruct.ground_truth_utils import VerifierConfig  # noqa: PLC0415

    base_names = {f.name for f in dataclasses.fields(StreamingDataLoaderConfig)}
    new_fields: list[tuple[str, type, dataclasses.Field]] = []
    seen: set[str] = set()

    for cfg_cls in _registered_verifier_config_classes:
        if not issubclass(cfg_cls, VerifierConfig):
            raise TypeError(f"Expected VerifierConfig subclass, got {cfg_cls}")
        for f in dataclasses.fields(cfg_cls):
            if f.name in base_names or f.name in seen:
                continue
            seen.add(f.name)
            new_fields.append((f.name, f.type, f))

    return dataclasses.make_dataclass(class_name, new_fields, bases=(StreamingDataLoaderConfig,), frozen=False)
