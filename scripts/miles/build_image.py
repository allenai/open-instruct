"""Build the MILES runtime or application image against the checked-in runtime lock.
The command verifies the local base-image identity, supplies authenticated source
access through a BuildKit secret when needed, and records the application revision
in the build arguments. The committed-image launcher uses this to package Open
Instruct with its separately pinned GPU dependencies.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target", choices=("runtime-base", "application"), default="application")
    parser.add_argument(
        "--source-cache",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Build from local committed sources; supply every locked repository",
    )
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    lock = json.loads((root / "runtime/miles/runtime.lock.json").read_text())
    actual = subprocess.check_output(
        ["docker", "image", "inspect", options.base_image, "--format", "{{.Id}}"], text=True
    ).strip()
    if actual != lock["base_image"]["docker_id"]:
        raise ValueError("Base image differs from runtime.lock.json; qualify a new runtime before changing the pin")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    env = dict(os.environ)
    secrets = []
    caches = dict(item.split("=", 1) for item in options.source_cache)
    if caches and set(caches) != set(lock["sources"]):
        raise ValueError("Local builds require one --source-cache for every locked repository")
    if not caches and any(source.get("private") for source in lock["sources"].values()):
        token = env.get("GH_TOKEN") or env.get("GITHUB_TOKEN")
        if not token:
            try:
                token = subprocess.check_output(["gh", "auth", "token"], text=True, stderr=subprocess.PIPE).strip()
            except (FileNotFoundError, subprocess.CalledProcessError) as error:
                raise ValueError("Private runtime sources require GH_TOKEN, GITHUB_TOKEN, or gh auth login") from error
        if not token:
            raise ValueError("GitHub authentication returned an empty token")
        env["MILES_BUILD_GITHUB_TOKEN"] = token
        secrets = ["--secret", "id=github_token,env=MILES_BUILD_GITHUB_TOKEN"]
    command = [
        "docker",
        "build",
        "--file",
        "runtime/miles/Dockerfile",
        "--build-arg",
        f"BASE_IMAGE={options.base_image}",
        "--build-arg",
        f"SOURCE_REVISION={revision}",
        "--target",
        options.target,
        "--tag",
        options.tag,
        *secrets,
        ".",
    ]
    with tempfile.TemporaryDirectory(prefix="miles-build-sources-") as directory:
        if caches:
            destination = Path(directory)
            cache_args = [
                part for name, path in caches.items() for part in ("--cache", f"{name}={Path(path).resolve()}")
            ]
            subprocess.run(
                [sys.executable, "-m", "scripts.miles.prepare_runtime", directory, *cache_args], cwd=root, check=True
            )
            command[-1:-1] = ["--build-arg", "SOURCE_MODE=local", "--build-context", f"runtime_sources={destination}"]
        subprocess.run(command, cwd=root, check=True, env=env)


if __name__ == "__main__":
    main()
