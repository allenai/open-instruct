"""Replay measured Triton choices for matched-process numerical diagnostics."""

import ast
import importlib


class PinnedCache(dict):
    """Retain chosen configs and count their use during the scoring capture."""

    def __init__(self, previous, pinned, marker):
        super().__init__(previous | pinned)
        self.pinned = pinned
        self.marker = marker
        self.hits = dict.fromkeys(pinned, 0)

    def __getitem__(self, key):
        if key in self.pinned and self.marker.exists():
            self.hits[key] += 1
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key in self.pinned and str(value) != str(self.pinned[key]):
            raise RuntimeError(f"Attempted to replace pinned Triton configuration: {key}")
        super().__setitem__(key, value)


def install(document, marker):
    """Resolve recorded choices against this runtime's actual candidate configs."""
    installed = {}
    for name, entries in document.items():
        module, attribute = name.rsplit(".", 1)
        if not module.startswith("fla.ops."):
            raise ValueError(f"Expected FLA kernel, got {name}")
        value = getattr(importlib.import_module(module), attribute)
        for _ in range(5):
            if isinstance(getattr(value, "cache", None), dict) and hasattr(value, "configs"):
                break
            value = getattr(value, "fn", None)
        else:
            raise ValueError(f"No autotuner found for {name}")
        choices = {str(config): config for config in value.configs}
        pinned = {}
        for raw_key, chosen in entries.items():
            key = ast.literal_eval(raw_key)
            if not isinstance(key, tuple) or chosen not in choices:
                raise ValueError(f"Recorded Triton choice is unavailable for {name}: {chosen}")
            pinned[key] = choices[chosen]
        value.cache = PinnedCache(value.cache, pinned, marker)
        installed[name] = value.cache
    if not installed:
        raise ValueError("Empty tuning replay")
    return installed


def report(installed):
    result = {}
    for name, cache in installed.items():
        entries = {}
        for key, config in cache.pinned.items():
            if str(dict.__getitem__(cache, key)) != str(config) or not cache.hits[key]:
                raise RuntimeError(f"Pinned config was changed or never used during scoring: {name} {key}")
            entries[repr(key)] = {"config": str(config), "scoring_hits": cache.hits[key]}
        result[name] = entries
    return result
