"""Read-only bounded inventory of supplied hero checkpoint paths and their parents."""

import argparse
import json
from pathlib import Path


def describe(path):
    result = {"path": str(path), "exists": path.exists(), "is_directory": path.is_dir()}
    if path.is_dir():
        children = sorted(path.iterdir())
        result["child_count"] = len(children)
        result["children"] = [
            {"name": child.name, "directory": child.is_dir(), "bytes": child.stat().st_size}
            for child in children[:200]
        ]
        result["truncated"] = len(children) > 200
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    mount = Path("/weka/olmo-3p5-checkpoints")
    paths = {mount, args.native, args.hf, args.hf.parent / "olmo-core"}
    for source in (args.native, args.hf):
        paths.update(parent for parent in source.parents if parent.is_relative_to(mount))
    result = {
        "mount_exists": mount.exists(),
        "is_mount": mount.is_mount(),
        "paths": [describe(path) for path in sorted(paths)],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
