"""Materialize runtime sources at their pinned Git commits.

Use --cache NAME=/existing/repo to reproduce from a local Git object cache.
No packages are installed or existing checkouts modified.
"""

import argparse
import base64
import json
import os
import subprocess
from pathlib import Path


def run(*args, cwd=None, env=None):
    try:
        return subprocess.run(args, cwd=cwd, env=env, check=True, text=True, capture_output=True).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Runtime source command failed: {exc.stderr.strip()}") from exc


def fetch_environment(token_file):
    """Authenticate GitHub fetches without putting credentials in URLs or Git config."""
    if token_file is None:
        return None
    token = token_file.read_text().strip()
    if not token:
        raise ValueError("GitHub token file is empty")
    header = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = dict(os.environ)
    index = int(env.get("GIT_CONFIG_COUNT", "0"))
    env.update(
        {
            "GIT_CONFIG_COUNT": str(index + 1),
            f"GIT_CONFIG_KEY_{index}": "http.https://github.com/.extraheader",
            f"GIT_CONFIG_VALUE_{index}": f"AUTHORIZATION: basic {header}",
        }
    )
    return env


def prepare_source(name, source, target, *, cache=None, token_file=None):
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
        env = fetch_environment(token_file) if source.get("private") else None
        run("git", "fetch", "--depth=1", source["repository"], source["revision"], cwd=target, env=env)
        run("git", "checkout", "--detach", "FETCH_HEAD", cwd=target)
    if run("git", "rev-parse", "HEAD", cwd=target) != source["revision"]:
        raise ValueError(f"Source revision mismatch for {name}")
    print(f"Prepared {name} at {source['revision']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--cache", action="append", default=[])
    parser.add_argument("--github-token-file", type=Path, help="Temporary credential file for private GitHub sources")
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    lock = json.loads((root / "runtime/miles/runtime.lock.json").read_text())
    caches = dict(item.split("=", 1) for item in options.cache)
    options.destination.mkdir(parents=True, exist_ok=True)
    for name, source in lock["sources"].items():
        prepare_source(
            name, source, options.destination / name, cache=caches.get(name), token_file=options.github_token_file
        )
    (options.destination / "runtime.lock.json").write_text(json.dumps(lock, indent=2) + "\n")


if __name__ == "__main__":
    main()
