"""Build the dependency-only MILES image from immutable, pre-qualified inputs.

Pull both Beaker images recorded in base-build.lock.json before invoking this
maintainer command. Ordinary application builds use build_image.py instead.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-parent", required=True)
    parser.add_argument("--flash-attention-builder", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-jobs", type=int, default=2)
    options = parser.parse_args()
    if options.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    root = Path(__file__).resolve().parents[2]
    lock = json.loads((root / "runtime/miles/base-build.lock.json").read_text())
    for key, image in (
        ("compiled_parent", options.compiled_parent),
        ("flash_attention_builder", options.flash_attention_builder),
    ):
        actual = subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
        ).strip()
        if actual != lock[key]["docker_id"]:
            raise ValueError(f"{key} differs from base-build.lock.json")
    subprocess.run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "--file",
            "runtime/miles/Dockerfile.base",
            "--build-arg",
            f"COMPILED_PARENT={options.compiled_parent}",
            "--build-arg",
            f"COMPILED_PARENT_DIGEST={lock['compiled_parent']['docker_id']}",
            "--build-arg",
            f"FLASH_ATTENTION_BUILDER={options.flash_attention_builder}",
            "--build-arg",
            f"FLASH_ATTENTION_MAX_JOBS={options.max_jobs}",
            "--tag",
            options.tag,
            ".",
        ],
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    main()
