"""Build the Core RL source overlay on the exact runtime recorded in the lock."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target", choices=("runtime-base", "application"), default="application")
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
    if any(source.get("private") for source in lock["sources"].values()):
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
    subprocess.run(
        [
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
        ],
        cwd=root,
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
