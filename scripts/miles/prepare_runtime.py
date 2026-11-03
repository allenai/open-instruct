"""Materialize the runtime lock's source repositories at their exact Git commits.
Image builds use these checkouts to install the selected MILES, Core and serving
implementations, with optional local Git caches to reuse existing objects. The
script prepares sources only; it neither installs packages nor modifies the cache
repositories supplied by the caller.

Use --cache NAME=/existing/repo to reproduce from a local Git object cache.
No packages are installed or existing checkouts modified.
"""

import argparse
import json
import subprocess
from pathlib import Path


def run(*args, cwd=None):
    try:
        return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Runtime source command failed: {exc.stderr.strip()}") from exc


def prepare_source(name, source, target, *, cache=None):
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing source tree: {target}")
    if cache is not None:
        run("git", "clone", "--no-checkout", "--shared", str(cache), str(target))
        run("git", "checkout", "--detach", source["revision"], cwd=target)
        # A runtime image must not depend on the cache's Git alternates file.
        run("git", "repack", "-a", "-d", cwd=target)
        (target / ".git/objects/info/alternates").unlink(missing_ok=True)
    else:
        run("git", "init", str(target))
        run("git", "fetch", "--depth=1", source["repository"], source["revision"], cwd=target)
        run("git", "checkout", "--detach", "FETCH_HEAD", cwd=target)
    if run("git", "rev-parse", "HEAD", cwd=target) != source["revision"]:
        raise ValueError(f"Source revision mismatch for {name}")
    print(f"Prepared {name} at {source['revision']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--cache", action="append", default=[])
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    lock = json.loads((root / "runtime/miles/runtime.lock.json").read_text())
    caches = dict(item.split("=", 1) for item in options.cache)
    options.destination.mkdir(parents=True, exist_ok=True)
    for name, source in lock["sources"].items():
        prepare_source(name, source, options.destination / name, cache=caches.get(name))
    (options.destination / "runtime.lock.json").write_text(json.dumps(lock, indent=2) + "\n")


if __name__ == "__main__":
    main()
