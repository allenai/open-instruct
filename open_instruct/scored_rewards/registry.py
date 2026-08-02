"""Plugin loading, so your reward lives in your repo and not in this one.

open-instruct discovers verifiers by walking ``VerifierFunction.__subclasses__()``,
which means a verifier only exists if some module that defines it has already
been imported - in practice, if you edited ``ground_truth_utils.py``. That is
fine for a verifier everyone shares and wrong for a reward that belongs to one
experiment.

``--reward_plugins`` names modules to import before anything is built. A plugin
is any importable module, or a path to a ``.py`` file, that calls ``register``
at import time. Importing it is the whole protocol: registering a scorer,
adding an environment to ``TOOL_REGISTRY``, and defining a ``VerifierFunction``
subclass all happen as a side effect of the import, and all three are picked up.

    --reward_plugins projects.tutor.plugin
    --reward_plugins projects/tutor/plugin.py,my_other_rewards.py
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from collections.abc import Callable
from typing import Any

from open_instruct.scored_rewards.types import FunctionScorer, GroupScorer, PerSample, Scorer

logger = logging.getLogger(__name__)

#: name -> factory. A factory takes keyword arguments and returns a Scorer or
#: GroupScorer; a bare class is itself a factory.
_FACTORIES: dict[str, Callable[..., Any]] = {}
_LOADED_PLUGINS: set[str] = set()


def register(name: str, factory: Callable[..., Any] | None = None):
    """Register a scorer factory under ``name``. Usable as a decorator.

    @register("my_reward")
    class MyReward(Scorer):
        ...

    @register("my_reward")
    def build(threshold: float = 0.5) -> Scorer:
        ...
    """

    def _apply(f: Callable[..., Any]) -> Callable[..., Any]:
        key = name.lower()
        if key in _FACTORIES and _FACTORIES[key] is not f:
            logger.warning("scored_rewards: re-registering %r (was %r)", key, _FACTORIES[key])
        _FACTORIES[key] = f
        return f

    return _apply if factory is None else _apply(factory)


def register_fn(name: str):
    """Register a plain ``(Sample) -> float`` callable as a per-sample scorer."""

    def _apply(fn):
        register(name, lambda **kw: FunctionScorer(fn, name=name))
        return fn

    return _apply


def available() -> list[str]:
    return sorted(_FACTORIES)


def build(spec: str, **overrides) -> GroupScorer:
    """Build a group scorer from a spec string.

    ``spec`` is ``name`` or ``name:key=value,key=value``. Values are parsed as
    JSON when they parse and left as strings when they do not, so
    ``tutor:turns=3,judge_model=Qwen/Qwen2.5-7B-Instruct`` does what it looks
    like. A per-sample ``Scorer`` is lifted into a group scorer automatically.
    """
    name, _, arg_string = spec.partition(":")
    key = name.strip().lower()
    if key not in _FACTORIES:
        raise KeyError(f"no scorer registered as {key!r}. Registered: {available()}. Did you pass --reward_plugins?")

    kwargs = dict(parse_kwargs(arg_string))
    kwargs.update(overrides)
    built = _FACTORIES[key](**kwargs)
    if isinstance(built, GroupScorer):
        return built
    if isinstance(built, Scorer):
        return PerSample(built, name=key)
    raise TypeError(f"factory {key!r} returned {type(built).__name__}, expected Scorer or GroupScorer")


def parse_kwargs(arg_string: str) -> dict[str, Any]:
    """``a=1,b=hello,c=[1,2]`` -> ``{"a": 1, "b": "hello", "c": [1, 2]}``."""
    import json  # noqa: PLC0415

    out: dict[str, Any] = {}
    depth = 0
    current = ""
    parts: list[str] = []
    # split on commas that are not inside a JSON literal
    for ch in arg_string:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        k, sep, v = part.partition("=")
        if not sep:
            raise ValueError(f"scorer argument {part!r} is not key=value")
        v = v.strip()
        try:
            out[k.strip()] = json.loads(v)
        except (json.JSONDecodeError, ValueError):
            out[k.strip()] = v
    return out


def load_plugins(spec: str | None) -> list[str]:
    """Import every module named in a comma-separated ``spec``.

    Accepts dotted module paths and filesystem paths to ``.py`` files. Idempotent:
    the same plugin named twice is imported once, which matters because this is
    called from more than one process in a Ray job.
    """
    if not spec:
        return []
    loaded = []
    for raw in spec.split(","):
        target = raw.strip()
        if not target or target in _LOADED_PLUGINS:
            continue
        _import_one(target)
        _LOADED_PLUGINS.add(target)
        loaded.append(target)
    if loaded:
        logger.info("scored_rewards: loaded plugins %s; scorers now %s", loaded, available())
    return loaded


def _import_one(target: str) -> None:
    if target.endswith(".py") or os.sep in target:
        path = os.path.abspath(target)
        if not os.path.exists(path):
            raise FileNotFoundError(f"reward plugin {target!r} not found at {path}")
        module_name = "scored_rewards_plugin_" + os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load reward plugin from {path}")
        module = importlib.util.module_from_spec(spec)
        # register before exec so a plugin that imports itself does not recurse
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return
    importlib.import_module(target)
