"""CPU-safe encoding using a snapshot of the pinned MILES/SGLang argparse contract.

Regenerate with scripts/miles/snapshot_options.py inside the pinned runtime.
Parser support does not establish that a particular backend implements an option.
"""

import argparse
import difflib
import json
import math
from functools import lru_cache
from pathlib import Path

from open_instruct.miles import validation
from open_instruct.miles.errors import InputError


def describe_parser(parser):
    records = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        kind = "value"
        if isinstance(action, argparse.BooleanOptionalAction):
            kind = "boolean"
        elif (
            isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))
            or type(action).__name__ == "DeprecatedStoreTrueAction"
        ):
            kind = "switch"
        elif isinstance(action, argparse._AppendAction):
            kind = "append"
        elif type(action).__name__ == "DeprecatedStoreConstAction":
            kind = "alias_const"
        elif not isinstance(action, argparse._StoreAction) and type(action).__name__ != "DeprecatedAliasStoreAction":
            kind = "unsupported"
        record = {
            "dest": action.dest,
            "flags": action.option_strings,
            "kind": kind,
            "nargs": action.nargs,
            "type": getattr(action.type, "__name__", None),
        }
        if action.choices is not None:
            record["choices"] = sorted(action.choices, key=str)
        if kind in ("boolean", "switch"):
            record.update(default=action.default, const=action.const)
        if kind == "alias_const":
            record["const"] = action.const_value
        records.append(record)
    return records


@lru_cache(maxsize=1)
def option_index():
    records = json.loads(Path(__file__).with_name("options.json").read_text())["options"]
    index = {}
    for record in records:
        canonical_flag = "--" + record["dest"].replace("_", "-")
        if record["dest"] not in index or canonical_flag in record["flags"]:
            index[record["dest"]] = record
    for record in records:
        for flag in record["flags"]:
            index.setdefault(flag.removeprefix("--").replace("-", "_"), record)
    return index


def resolve_option(name, value):
    index = option_index()
    if not isinstance(name, str):
        raise InputError("MILES option names must be strings; use underscores, for example global_batch_size.")
    if name not in index:
        close = difflib.get_close_matches(name, index, n=1)
        hint = f"; did you mean {close[0]}?" if close else ""
        raise InputError(f"Unknown MILES option: {name}{hint}")
    record = index[name]
    if record["kind"] == "alias_const":
        validation.boolean(value, f"miles.{name}")
        value = record["const"] if value else None
    elif record["kind"] in ("boolean", "switch"):
        validation.boolean(value, f"miles.{name}")
        if name != record["dest"]:
            flag = "--" + name.replace("_", "-")
            positive = not flag.startswith("--no-") if record["kind"] == "boolean" else record["const"]
            value = positive if value else not positive
    return record, value


def normalize_options(options):
    validation.mapping(options, "[miles]")
    result = {}
    seen = set()
    for name, value in options.items():
        record, value = resolve_option(name, value)
        dest = record["dest"]
        if dest in seen:
            raise InputError(f"Multiple spellings supplied for miles.{dest}; use one option name")
        seen.add(dest)
        if record["kind"] != "alias_const" or value is not None:
            result[dest] = value
    return result


def _scalar(record, value):
    kind = record["type"]
    valid = isinstance(value, (str, int, float)) and not isinstance(value, bool)
    if kind == "int":
        valid = type(value) is int
    elif kind == "float":
        valid = type(value) in (int, float)
    elif kind in ("str", "nullable_str"):
        valid = isinstance(value, str)
    if not valid or (isinstance(value, float) and not math.isfinite(value)):
        raise InputError(f"miles.{record['dest']} expects {kind or 'a scalar'}, got {value!r}")
    if "choices" in record and value not in record["choices"]:
        raise InputError(f"miles.{record['dest']} must be one of {record['choices']}")
    return str(value)


def encode_options(options):
    result = []
    for name, value in normalize_options(options).items():
        record = option_index()[name]
        flag = record["flags"][0]
        kind = record["kind"]
        if kind == "unsupported":
            raise InputError(f"miles.{name} uses a custom/deprecated parser action; use its current native option")
        if kind == "boolean":
            result.append(flag if value else next(f for f in record["flags"] if f.startswith("--no-")))
        elif kind == "switch":
            if value == record["const"]:
                result.append(flag)
            elif value != record["default"]:
                raise InputError(f"miles.{name} cannot represent {value} with the pinned parser")
        elif record["type"] in ("loads", "json_list_type", "parse_cuda_graph_config_arg") or (
            name.endswith("json_model_override_args")
        ):
            # Both native JSON strings and structured TOML values are accepted.
            try:
                parsed = json.loads(value) if isinstance(value, str) else value
                encoded = json.dumps(parsed, allow_nan=False, separators=(",", ":"))
            except (ValueError, TypeError) as error:
                raise InputError(
                    f"miles.{name} must contain valid finite JSON; use a TOML inline table or a quoted JSON string."
                ) from error
            if record["type"] == "json_list_type" and not isinstance(parsed, list):
                raise InputError(f"miles.{name} expects a JSON list; use [value1, value2].")
            result.extend([flag, encoded])
        else:
            values = value if kind == "append" else [value]
            if kind == "append" and (not isinstance(values, list) or not values):
                raise InputError(f"miles.{name} expects a nonempty list of occurrences")
            for occurrence in values:
                nargs = record["nargs"]
                if nargs in ("+", "*") or isinstance(nargs, int):
                    if not isinstance(occurrence, list):
                        raise InputError(f"miles.{name} expects a list")
                    if (nargs == "+" and not occurrence) or (isinstance(nargs, int) and len(occurrence) != nargs):
                        raise InputError(
                            f"Invalid number of values for miles.{name}: expected {nargs if isinstance(nargs, int) else 'at least one'}, got {len(occurrence)}."
                        )
                    result.extend([flag, *[_scalar(record, item) for item in occurrence]])
                else:
                    # Equals form also protects string values beginning with '--'.
                    encoded = _scalar(record, occurrence)
                    if encoded.startswith("-"):
                        result.append(f"{flag}={encoded}")
                    else:
                        result.extend([flag, encoded])
    return result
