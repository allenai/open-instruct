"""CLI argument parsing built on tyro.

Provides a drop-in replacement for the previous ``ArgumentParserPlus`` (which
wrapped ``transformers.HfArgumentParser``). The public entry point is
``parse``, which accepts one or more dataclass types and returns parsed
instances. Compared to the HF parser, tyro gives real static-type-checked
parsing, native support for ``Literal``/``Union``/``Enum``/nested dataclasses,
and removes the need for ``assert isinstance`` after parsing.

The CLI surface is kept flat (``--foo`` rather than ``--group.foo``) via
``tyro.conf.OmitArgPrefixes`` so existing launch scripts keep working.

YAML configs are supported via ``parse``: if the first positional argument on
the command line is a ``.yaml`` path, values are loaded from that file and the
remaining flags override them. This mirrors the old behaviour.
"""

import dataclasses
import os
import sys
from typing import Any

import yaml
from tyro import cli as tyro_cli
from tyro import conf as tyro_conf

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def _make_container(dataclass_types: tuple[type, ...]) -> type:
    """Build a synthetic dataclass whose fields are the given dataclasses.

    Each field is required (no default) so tyro can derive defaults from each
    inner dataclass without us having to instantiate it (which would fail for
    dataclasses with required fields).
    """
    fields = []
    seen = {}
    for i, dc in enumerate(dataclass_types):
        if not dataclasses.is_dataclass(dc):
            raise TypeError(f"Expected a dataclass type, got {dc!r}")
        base = getattr(dc, "__name__", f"dc_{i}").lower()
        name = base
        suffix = 1
        while name in seen:
            suffix += 1
            name = f"{base}_{suffix}"
        seen[name] = True
        fields.append((name, dc))
    return dataclasses.make_dataclass("_ParsedContainer", fields)


def _apply_yaml_defaults(dataclass_types: tuple[type, ...], yaml_path: str) -> tuple:
    """Return instances of ``dataclass_types`` with values from a YAML file.

    Keys are matched to fields by name across all dataclasses. Unknown keys
    raise ``ValueError`` unless ``allow_extra_keys`` is set on the caller.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data).__name__} in {yaml_path}")
    return data


def _field_map(dataclass_types: tuple[type, ...]) -> dict[str, tuple[int, dataclasses.Field]]:
    """Build a mapping from field name -> (dataclass index, Field)."""
    out: dict[str, tuple[int, dataclasses.Field]] = {}
    for i, dc in enumerate(dataclass_types):
        for f in dataclasses.fields(dc):
            if not f.init:
                continue
            if f.name in out:
                raise ValueError(
                    f"Duplicate field name {f.name!r} across dataclasses "
                    f"{dataclass_types[out[f.name][0]].__name__} and {dc.__name__}"
                )
            out[f.name] = (i, f)
    return out


def _yaml_dict_to_cli_args(
    yaml_data: dict[str, Any], field_map: dict[str, tuple[int, dataclasses.Field]], allow_extra_keys: bool
) -> list[str]:
    """Convert a YAML dict into a list of ``--key value`` CLI tokens.

    We funnel YAML values through tyro by re-encoding them as CLI flags. This
    keeps a single parsing/casting code path (tyro) instead of duplicating type
    coercion as the old ``parse_yaml_and_args`` did.
    """
    tokens: list[str] = []
    for key, value in yaml_data.items():
        if key not in field_map:
            if allow_extra_keys:
                logger.warning("Ignoring extra YAML key %r (not present in any dataclass)", key)
                continue
            raise ValueError(f"YAML key {key!r} does not match any dataclass field")
        flag = f"--{key}"
        if isinstance(value, bool):
            tokens.append(flag if value else f"--no-{key}")
        elif isinstance(value, (list, tuple)):
            tokens.append(flag)
            tokens.extend(str(v) for v in value)
        elif value is None:
            tokens.append(flag)
            tokens.append("None")
        else:
            tokens.append(flag)
            tokens.append(str(value))
    return tokens


def parse(
    *dataclass_types: type,
    args: list[str] | None = None,
    defaults: dict[str, Any] | None = None,
    allow_extra_keys: bool = False,
) -> Any:
    """Parse CLI arguments into instances of ``dataclass_types``.

    Args:
        *dataclass_types: One or more dataclass types to populate.
        args: Optional argv to parse; defaults to ``sys.argv[1:]``.
        defaults: Optional mapping of field name -> default override. Applied
            before parsing, like the old ``parser.set_defaults(...)``. Each
            key must correspond to exactly one field across all dataclasses.
        allow_extra_keys: If ``True``, ignore YAML keys that don't match any
            field instead of raising.

    Returns:
        A single dataclass instance if ``len(dataclass_types) == 1``, otherwise
        a tuple of instances in the order given.
    """
    if not dataclass_types:
        raise ValueError("parse() requires at least one dataclass type")

    raw_args = list(args) if args is not None else list(sys.argv[1:])
    field_map = _field_map(dataclass_types)

    yaml_tokens: list[str] = []
    if raw_args and raw_args[0].endswith(".yaml") and os.path.isfile(raw_args[0]):
        yaml_data = _apply_yaml_defaults(dataclass_types, raw_args.pop(0))
        yaml_tokens = _yaml_dict_to_cli_args(yaml_data, field_map, allow_extra_keys)

    if defaults:
        for key in defaults:
            if key not in field_map:
                raise ValueError(f"defaults key {key!r} does not match any dataclass field")

    container_cls = _make_container(dataclass_types)
    inner_names = [f.name for f in dataclasses.fields(container_cls)]

    default_instance = None
    if defaults:
        inner_defaults: dict[int, dict[str, Any]] = {}
        for key, value in defaults.items():
            idx, _ = field_map[key]
            inner_defaults.setdefault(idx, {})[key] = value
        instances = []
        for i, dc in enumerate(dataclass_types):
            try:
                instances.append(dc(**inner_defaults.get(i, {})))
            except TypeError:
                if i in inner_defaults:
                    raise
                instances.append(None)
        if all(inst is not None for inst in instances):
            default_instance = container_cls(**dict(zip(inner_names, instances)))

    parsed = tyro_cli(
        container_cls, args=yaml_tokens + raw_args, config=(tyro_conf.OmitArgPrefixes,), default=default_instance
    )
    results = tuple(getattr(parsed, name) for name in inner_names)
    if len(results) == 1:
        return results[0]
    return results
