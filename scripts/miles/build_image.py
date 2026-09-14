"""Build the Core RL source overlay on the exact runtime recorded in the lock."""

import argparse
import json
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
            ".",
        ],
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    main()
