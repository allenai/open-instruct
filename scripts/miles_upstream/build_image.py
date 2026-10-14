"""Build upstream MILES from an immutable image/source pair, or refresh that pair."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = ROOT / "runtime/miles-upstream/runtime.lock.json"
REPOSITORY = "https://github.com/radixark/miles.git"
LABEL = "ai2.miles-upstream.build-id"
HELPERS = (
    "open_instruct/__init__.py",
    "open_instruct/miles/__init__.py",
    "open_instruct/miles/errors.py",
    "open_instruct/miles/configuration/__init__.py",
    "open_instruct/miles/configuration/options.py",
    "open_instruct/miles/configuration/validation.py",
)


def output(command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs).strip()


def read_lock(path):
    lock = json.loads(Path(path).read_text())
    if lock.get("schema_version") != 1 or lock.get("platform") != "linux/amd64":
        raise ValueError("Expected schema_version=1 and platform=linux/amd64")
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", lock.get("base_image", "")):
        raise ValueError("base_image must use an immutable registry digest")
    source = lock.get("miles", {})
    if source.get("repository") != REPOSITORY or not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", "")):
        raise ValueError("miles must select the public radixark/miles repository and a full commit SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", lock.get("image_inputs", "")):
        raise ValueError("image_inputs must be the upstream dependency-build fingerprint")
    return lock


def build_inputs(lock):
    files = {
        "Dockerfile": (ROOT / "runtime/miles-upstream/Dockerfile").read_bytes(),
        "runtime.lock.json": (json.dumps(lock, sort_keys=True, indent=2) + "\n").encode(),
    }
    for relative in HELPERS:
        files[f"adapter/{relative}"] = (ROOT / relative).read_bytes()
    for path in sorted((ROOT / "scripts/miles_upstream").glob("*.py")):
        files[f"adapter/scripts/miles_upstream/{path.name}"] = path.read_bytes()
    return files


def fingerprint(files):
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode() + b"\0" + hashlib.sha256(value).digest())
    return digest.hexdigest()


def inspect_registry(reference, field):
    return json.loads(
        output(["docker", "buildx", "imagetools", "inspect", reference, "--format", "{{json ." + field + "}}"])
    )


def refresh(path, reference, image):
    """Resolve moving inputs once; never update a lock with mismatched dependencies."""
    manifest = inspect_registry(image, "Manifest")
    if "manifests" in manifest:
        candidates = [
            item["digest"]
            for item in manifest["manifests"]
            if item.get("platform", {}).get("os") == "linux" and item["platform"].get("architecture") == "amd64"
        ]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one linux/amd64 image")
        digest = candidates[0]
    else:
        digest = manifest["digest"]
    repository = image.split("@")[0]
    # Strip a tag, preserving an optional registry port.
    if ":" in repository.rsplit("/", 1)[-1]:
        repository = repository.rsplit(":", 1)[0]
    pinned = f"{repository}@{digest}"
    metadata = inspect_registry(pinned, "Image")
    if metadata.get("architecture") != "amd64" or metadata.get("os") != "linux":
        raise ValueError("The selected image is not linux/amd64")
    expected = metadata.get("config", {}).get("Labels", {}).get("miles.image-inputs")
    with tempfile.TemporaryDirectory(prefix="miles-upstream-source-") as directory:
        subprocess.run(["git", "init", directory], check=True, capture_output=True)
        subprocess.run(["git", "fetch", "--depth=1", REPOSITORY, reference], cwd=directory, check=True)
        subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=directory, check=True)
        revision = output(["git", "rev-parse", "HEAD"], cwd=directory)
        actual = output(["python3", "docker/image_inputs.py"], cwd=directory)
    if not expected or actual != expected:
        raise ValueError("Upstream dependency inputs differ from the image; build its upstream Docker recipe first")
    lock = {
        "schema_version": 1,
        "qualification": "candidate: GPU lifecycle validation pending",
        "platform": "linux/amd64",
        "base_image": pinned,
        "image_inputs": actual,
        "miles": {"repository": REPOSITORY, "revision": revision},
    }
    path.write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps(lock, indent=2))


def command(lock, build_id, tag, target, directory):
    return [
        "docker",
        "buildx",
        "build",
        "--platform",
        lock["platform"],
        "--load",
        "--build-arg",
        f"BASE_IMAGE={lock['base_image']}",
        "--build-arg",
        f"MILES_REVISION={lock['miles']['revision']}",
        "--build-arg",
        f"IMAGE_INPUTS={lock['image_inputs']}",
        "--build-arg",
        f"BUILD_ID={build_id}",
        "--target",
        target,
        "--tag",
        tag,
        directory,
    ]


def build(args):
    lock = read_lock(args.lock)
    files = build_inputs(lock)
    build_id = fingerprint(files)
    tag = args.tag or f"miles-upstream-{args.target}-{build_id[:16]}"
    if args.dry_run:
        print(
            json.dumps(
                {"build_id": build_id, "command": command(lock, build_id, tag, args.target, "BUILD_CONTEXT")}, indent=2
            )
        )
        return
    if args.publish_workspace and output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("Commit changes before publishing an image to Beaker")
    metadata = inspect_registry(lock["base_image"], "Image")
    if metadata.get("config", {}).get("Labels", {}).get("miles.image-inputs") != lock["image_inputs"]:
        raise ValueError("The base image's dependency fingerprint differs from the lock")
    with tempfile.TemporaryDirectory(prefix="miles-upstream-build-") as directory:
        for name, content in files.items():
            target = Path(directory) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        subprocess.run(command(lock, build_id, tag, args.target, directory), check=True)
    destination = args.output_dir or ROOT / "runs/miles-upstream-build" / build_id
    destination.mkdir(parents=True, exist_ok=True)
    receipt = {"build_id": build_id, "docker_tag": tag, "target": args.target, "lock": lock}
    receipt["docker_id"] = output(["docker", "image", "inspect", tag, "--format", "{{.Id}}"])
    if args.target == "application":
        container = output(["docker", "create", tag])
        try:
            # A fresh destination prevents docker cp from nesting schemas on repeat builds.
            with tempfile.TemporaryDirectory(prefix="miles-upstream-schemas-") as directory:
                subprocess.run(["docker", "cp", f"{container}:/opt/miles-upstream/schemas", directory], check=True)
                shutil.copytree(Path(directory) / "schemas", destination / "schemas", dirs_exist_ok=True)
        finally:
            subprocess.run(["docker", "rm", container], check=True)
    if args.publish_workspace:
        response = output(
            [
                "beaker",
                "image",
                "create",
                tag,
                "--name",
                f"miles-upstream-{args.target}-{build_id[:16]}",
                "--workspace",
                args.publish_workspace,
                "--description",
                f"Upstream MILES; build-id={build_id}",
                "--quiet",
            ]
        )
        # `image create` ignores --format; --quiet prints the immutable ID.
        if not re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", response):
            raise ValueError(f"Expected a Beaker image ID from image create --quiet, got {response!r}")
        receipt["beaker"] = response
    (destination / "image.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    print(f"Build artifacts: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    refresh_parser = subparsers.add_parser("refresh")
    refresh_parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    refresh_parser.add_argument("--ref", default="main")
    refresh_parser.add_argument("--base-image", default="radixark/miles:latest")
    builder = subparsers.add_parser("build")
    builder.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    builder.add_argument("--tag")
    builder.add_argument("--target", choices=("runtime", "application"), default="application")
    builder.add_argument("--dry-run", action="store_true")
    builder.add_argument("--output-dir", type=Path)
    builder.add_argument("--publish-workspace")
    args = parser.parse_args()
    if args.action == "refresh":
        refresh(args.lock, args.ref, args.base_image)
    else:
        build(args)


if __name__ == "__main__":
    main()
