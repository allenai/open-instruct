"""Build the MILES runtime or application image against the checked-in runtime lock.
The command verifies the local base-image identity, supplies authenticated source
access through a BuildKit secret when needed, and records the application revision
in the build arguments. The committed-image launcher uses this to package Open
Instruct with its separately pinned GPU dependencies.

``--runtime-base IMAGE`` instead builds only the application stage on a published
``runtime-base`` image (FlashAttention, locked sources and verifier packages). That
build compiles nothing and needs no private-source access; it refuses a runtime base
whose recorded inputs differ from this checkout's.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

RUNTIME_INPUTS_LABEL = "org.allenai.miles.runtime-inputs"
"""Image label recording ``runtime_inputs_digest`` of the checkout that built a runtime base."""
REVISION_LABEL = "org.opencontainers.image.revision"
"""Set only by the application stage, so its presence marks an application image."""
APPLICATION_STAGE = "FROM runtime-base AS application"


def runtime_inputs_digest(root):
    """Digest of the checked-in files that determine the ``runtime-base`` stage.

    Covers the runtime directory (lock, verifier requirements and the Dockerfile up to
    the application stage) and the source preparation script.
    """
    runtime = root / "runtime/miles"
    paths = sorted(
        path
        for path in runtime.rglob("*")
        if path.is_file() and "sources" not in path.relative_to(runtime).parts and "__pycache__" not in path.parts
    )
    paths.append(root / "scripts/miles/prepare_runtime.py")
    digest = hashlib.sha256()
    for path in paths:
        content = path.read_bytes()
        if path == runtime / "Dockerfile":
            content = content.split(APPLICATION_STAGE.encode())[0]
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def image_labels(image):
    output = subprocess.check_output(
        ["docker", "image", "inspect", image, "--format", "{{json .Config.Labels}}"], text=True
    )
    return json.loads(output) or {}


def github_token(env):
    token = env.get("GH_TOKEN") or env.get("GITHUB_TOKEN")
    if not token:
        try:
            token = subprocess.check_output(["gh", "auth", "token"], text=True, stderr=subprocess.PIPE).strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise ValueError("Private runtime sources require GH_TOKEN, GITHUB_TOKEN, or gh auth login") from error
    if not token:
        raise ValueError("GitHub authentication returned an empty token")
    return token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = parser.add_mutually_exclusive_group(required=True)
    base.add_argument("--base-image", help="Locally loaded binary base pinned in runtime.lock.json")
    base.add_argument("--runtime-base", help="Locally loaded runtime-base image; builds only the application stage")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target", choices=("runtime-base", "application"), default="application")
    parser.add_argument("--flash-attention-max-jobs", type=int, help="FA3 compile jobs (Dockerfile default: 2)")
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    lock = json.loads((root / "runtime/miles/runtime.lock.json").read_text())
    inputs = runtime_inputs_digest(root)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    env = dict(os.environ)
    build_args = ["--build-arg", f"SOURCE_REVISION={revision}"]
    if options.runtime_base:
        if options.target != "application":
            raise ValueError("--runtime-base builds the application stage only")
        labels = image_labels(options.runtime_base)
        if REVISION_LABEL in labels:
            raise ValueError("--runtime-base needs a runtime-base image, not an application image")
        if labels.get(RUNTIME_INPUTS_LABEL) != inputs:
            raise ValueError(
                f"Runtime base was built from different runtime inputs ({labels.get(RUNTIME_INPUTS_LABEL)}); "
                f"this checkout needs {inputs}. Build and publish a new runtime base."
            )
        build_args += ["--build-context", f"runtime-base=docker-image://{options.runtime_base}"]
    else:
        actual = subprocess.check_output(
            ["docker", "image", "inspect", options.base_image, "--format", "{{.Id}}"], text=True
        ).strip()
        if actual != lock["base_image"]["docker_id"]:
            raise ValueError(
                "Base image differs from runtime.lock.json; qualify a new runtime before changing the pin"
            )
        build_args += ["--build-arg", f"BASE_IMAGE={options.base_image}", "--build-arg", f"RUNTIME_INPUTS={inputs}"]
        if options.flash_attention_max_jobs is not None:
            build_args += ["--build-arg", f"FLASH_ATTENTION_MAX_JOBS={options.flash_attention_max_jobs}"]
        if any(source.get("private") for source in lock["sources"].values()):
            env["MILES_BUILD_GITHUB_TOKEN"] = github_token(env)
            build_args += ["--secret", "id=github_token,env=MILES_BUILD_GITHUB_TOKEN"]
    subprocess.run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "--file",
            "runtime/miles/Dockerfile",
            *build_args,
            "--target",
            options.target,
            "--tag",
            options.tag,
            ".",
        ],
        cwd=root,
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
