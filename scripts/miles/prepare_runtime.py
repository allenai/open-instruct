"""Materialize locked Core/MILES sources, checking every patch before applying it.

Use --cache NAME=/existing/repo to reproduce from a local Git object cache.
No packages are installed or existing checkouts modified.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def run(*args, cwd=None):
    try:
        return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Runtime source command failed: {exc.stderr.strip()}") from exc


def copy_embedded_source(source, target):
    """Reconstruct a private source overlay from the checksum-pinned base image.

    The base-image digest authenticates the tree; its marker must also match the
    requested source revision. A local Git baseline lets the normal full-index
    patch verification validate every touched file without network credentials.
    """
    embedded = Path(source["embedded_path"])
    if (embedded / ".source-revision").read_text().strip() != source["revision"]:
        raise ValueError(f"Embedded source revision mismatch: {embedded}")
    shutil.copytree(embedded, target, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    run("git", "init", str(target))
    run("git", "add", "--all", cwd=target)
    run(
        "git",
        "-c",
        "user.name=Runtime source snapshot",
        "-c",
        "user.email=runtime@localhost",
        "commit",
        "--quiet",
        "-m",
        f"Base image source {source['revision']}",
        cwd=target,
    )


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
        patch = root / "runtime/miles" / source["patch"]
        if hashlib.sha256(patch.read_bytes()).hexdigest() != source["patch_sha256"]:
            raise ValueError(f"Patch checksum mismatch for {name}")
        target = options.destination / name
        if target.exists():
            raise FileExistsError(f"Refusing to replace existing source tree: {target}")
        if name in caches:
            run("git", "clone", "--no-checkout", "--shared", caches[name], str(target))
            run("git", "checkout", "--detach", source["revision"], cwd=target)
            # A runtime image must not depend on the cache's Git alternates file.
            run("git", "repack", "-a", "-d", cwd=target)
            (target / ".git/objects/info/alternates").unlink(missing_ok=True)
        elif "embedded_path" in source:
            copy_embedded_source(source, target)
        else:
            run("git", "init", str(target))
            run("git", "fetch", "--depth=1", source["repository"], source["revision"], cwd=target)
            run("git", "checkout", "--detach", "FETCH_HEAD", cwd=target)
        if name in caches or "embedded_path" not in source:
            assert run("git", "rev-parse", "HEAD", cwd=target) == source["revision"]
        run("git", "apply", "--check", str(patch), cwd=target)
        run("git", "apply", "--index", str(patch), cwd=target)
        actual = subprocess.check_output(["git", "diff", "--cached", "--binary", "--full-index", "HEAD"], cwd=target)
        if hashlib.sha256(actual).hexdigest() != source["patch_sha256"]:
            raise ValueError(f"Reconstructed source delta differs from the locked patch for {name}")
        print(f"Prepared {name} at {source['revision']} + {source['patch_sha256']}")
    (options.destination / "runtime.lock.json").write_text(json.dumps(lock, indent=2) + "\n")


if __name__ == "__main__":
    main()
