"""Resolve reviewed local source names to the existing MILES mixture API.

The catalog and its verifier registries are trusted configuration, like a run
file. Hashes pin portable bundles; no dataset row can choose a factory/import.
No change to source sampling or run-file semantics is needed.
"""

import argparse
import hashlib
import json
from pathlib import Path

from open_instruct.miles.datasets import mixture, run_data
from open_instruct.miles.errors import InputError


def resolve(catalog_path, weights):
    path = Path(catalog_path).resolve()
    catalog = json.loads(path.read_text())
    if catalog.get("schema_version") != 1:
        raise InputError("Unsupported source catalog version")
    sources = []
    for name, fraction in weights.items():
        entry = catalog["sources"].get(name)
        if not entry or entry.get("status") != "qualified":
            raise InputError(f"Source {name!r} is not qualified in this catalog")
        source = {"name": name, "fraction": fraction}
        for key in ("prompt_data", "reward_config"):
            descriptor = entry[key]
            filename = (path.parent / descriptor["path"]).resolve()
            if not filename.is_relative_to(path.parent):
                raise InputError(f"Source {name}: bundle path must stay within the catalog directory")
            if hashlib.sha256(filename.read_bytes()).hexdigest() != descriptor["sha256"]:
                raise InputError(f"Source {name}: {key} hash mismatch")
            source[key] = str(filename)
        sources.append(source)
    mixture.validate_sources(sources)
    result = {"sources": sources, "seed": 17, "shuffle": True}
    run_data.validate_data(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--source", action="append", required=True, metavar="NAME=FRACTION")
    parser.add_argument("--output", type=Path, required=True, help="Write a [data] TOML fragment")
    args = parser.parse_args()
    weights = {}
    for item in args.source:
        name, value = item.rsplit("=", 1)
        if name in weights:
            parser.error(f"Repeated source {name}")
        weights[name] = float(value)
    data = resolve(args.catalog, weights)
    lines = ["[data]", "seed = 17", "shuffle = true", ""]
    for source in data["sources"]:
        lines.append("[[data.sources]]")
        lines.extend(f"{k} = {json.dumps(v)}" for k, v in source.items())
        lines.append("")
    with args.output.open("x") as output:
        output.write("\n".join(lines))


if __name__ == "__main__":
    main()
