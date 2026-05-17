#!/usr/bin/env python3
"""Warm a Docker Hub pull-through registry cache from a Tmax HF dataset.

The Tmax/SWERL datasets contain task directories with ``image.txt`` files,
usually packaged as ``task-data.tar.gz``. Tmax rows also carry per-sample
``env_config.image`` values. This script discovers both sources and pulls each
Docker Hub image through a registry mirror, causing the mirror to cache the
layers.

Examples:
    uv run python scripts/docker/warm_tmax_registry_mirror.py \\
        --dataset hamishivi/swerl-tmax-15k \\
        --mirror jupiter-cs-aus-193.reviz.ai2.in:5000

    uv run python scripts/docker/warm_tmax_registry_mirror.py \\
        --tool-configs '{"task_data_hf_repo":"hamishivi/swerl-tmax-15k","image":"python:3.12-slim"}' \\
        --mirror jupiter-cs-aus-193.reviz.ai2.in:5000

By default, the script warms the mirror through the registry HTTP API, so it
does not need a local Docker or Podman daemon. Use ``--method pull`` to warm
the mirror by running pulls like:
    docker pull jupiter-cs-aus-193.reviz.ai2.in:5000/hamishi740/swerl-tmax-v3:tag
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import tarfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm


MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)

_WARMED_BLOBS: set[str] = set()
_WARMED_BLOBS_LOCK = threading.Lock()


def resolve_task_data_dir(dataset: str, revision: str | None = None) -> Path:
    repo_dir = Path(snapshot_download(dataset, repo_type="dataset", revision=revision))
    tarball = repo_dir / "task-data.tar.gz"
    if not tarball.is_file():
        return repo_dir

    extract_dir = Path(f"{tarball}.extracted")
    marker = extract_dir / ".extract-complete"
    if marker.is_file():
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, mode="r:gz") as tar:
        tar.extractall(extract_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return extract_dir


def discover_images(task_data_dir: Path) -> list[str]:
    images: set[str] = set()
    for image_file in task_data_dir.rglob("image.txt"):
        image = image_file.read_text(encoding="utf-8").strip()
        if image:
            images.add(image)
    return sorted(images)


def _image_from_env_config(env_config) -> str | None:
    if isinstance(env_config, str):
        try:
            env_config = json.loads(env_config)
        except json.JSONDecodeError:
            return None
    if not isinstance(env_config, dict):
        return None
    image = env_config.get("image")
    return image if isinstance(image, str) and image else None


def discover_env_config_images(dataset: str, split: str, revision: str | None = None) -> list[str]:
    images: set[str] = set()
    rows_seen = 0
    ds = load_dataset(dataset, split=split, revision=revision, streaming=True)
    for row in ds:
        rows_seen += 1
        image = _image_from_env_config(row.get("env_config"))
        if image:
            images.add(image)
        if rows_seen % 1000 == 0:
            print(f"Scanned {rows_seen} dataset rows; found {len(images)} unique env_config images")
    print(f"Scanned {rows_seen} dataset rows; found {len(images)} unique env_config images")
    return sorted(images)


def load_tool_configs(raw: str | None) -> dict:
    if not raw:
        return {}

    maybe_path = Path(raw)
    if maybe_path.is_file():
        raw = maybe_path.read_text(encoding="utf-8")

    config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError("--tool-configs must be a JSON object or a path to a JSON object")
    return config


def _split_registry(image: str) -> tuple[str | None, str]:
    first, sep, rest = image.partition("/")
    if sep and ("." in first or ":" in first or first == "localhost"):
        return first, rest
    return None, image


def normalize_mirror(mirror: str) -> str:
    return re.sub(r"^https?://", "", mirror).rstrip("/")


def mirror_ref_for_dockerhub_image(image: str, mirror: str) -> str | None:
    registry, remainder = _split_registry(image)
    if registry is not None and registry not in {"docker.io", "registry-1.docker.io", "index.docker.io"}:
        return None

    if "/" not in remainder:
        remainder = f"library/{remainder}"

    return f"{normalize_mirror(mirror)}/{remainder}"


def dockerhub_repo_and_reference(image: str) -> tuple[str, str] | None:
    registry, remainder = _split_registry(image)
    if registry is not None and registry not in {"docker.io", "registry-1.docker.io", "index.docker.io"}:
        return None

    if "/" not in remainder:
        remainder = f"library/{remainder}"

    if "@" in remainder:
        repo, reference = remainder.rsplit("@", 1)
        return repo, reference

    last_slash = remainder.rfind("/")
    last_colon = remainder.rfind(":")
    if last_colon > last_slash:
        return remainder[:last_colon], remainder[last_colon + 1 :]

    return remainder, "latest"


def registry_url(mirror: str, path: str) -> str:
    return f"http://{normalize_mirror(mirror)}/v2/{path.lstrip('/')}"


def registry_get_json(mirror: str, path: str) -> tuple[dict, str]:
    request = urllib.request.Request(registry_url(mirror, path), headers={"Accept": MANIFEST_ACCEPT})
    with urllib.request.urlopen(request, timeout=300) as response:
        digest = response.headers.get("Docker-Content-Digest", "")
        return json.loads(response.read().decode("utf-8")), digest


def registry_get_blob(mirror: str, repo: str, digest: str, dry_run: bool) -> str:
    url = registry_url(mirror, f"{repo}/blobs/{digest}")
    with _WARMED_BLOBS_LOCK:
        if digest in _WARMED_BLOBS:
            return f"SKIP {url} (already requested)"
        _WARMED_BLOBS.add(digest)

    if dry_run:
        return f"GET {url}"

    request = urllib.request.Request(url)
    bytes_read = 0
    with urllib.request.urlopen(request, timeout=300) as response:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            bytes_read += len(chunk)
    return f"GET {url} ({bytes_read} bytes)"


def _manifest_matches_platform(manifest: dict, platform: str) -> bool:
    if platform == "all":
        return True
    expected = platform.split("/")
    actual = manifest.get("platform", {})
    if len(expected) < 2:
        raise ValueError("Platform must be 'os/arch', 'os/arch/variant', or 'all'")
    if actual.get("os") != expected[0] or actual.get("architecture") != expected[1]:
        return False
    if len(expected) >= 3 and actual.get("variant") != expected[2]:
        return False
    return True


def warm_with_registry_api(image: str, mirror: str, dry_run: bool, platform: str) -> tuple[str, bool, str]:
    parsed = dockerhub_repo_and_reference(image)
    if parsed is None:
        return image, True, "skipped non-Docker-Hub image"

    repo, reference = parsed
    try:
        if dry_run:
            return image, True, f"GET {registry_url(mirror, f'{repo}/manifests/{reference}')}"

        manifest, digest = registry_get_json(mirror, f"{repo}/manifests/{reference}")
        media_type = manifest.get("mediaType", "")
        logs = [f"manifest {reference} {digest}".strip()]

        if media_type in {
            "application/vnd.oci.image.index.v1+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
        }:
            manifest_digests = [
                item["digest"]
                for item in manifest.get("manifests", [])
                if item.get("digest") and _manifest_matches_platform(item, platform)
            ]
            if not manifest_digests:
                return image, False, f"No manifest in {reference} matched platform {platform}"
        else:
            manifest_digests = [digest or reference]

        blob_digests: set[str] = set()
        for manifest_digest in manifest_digests:
            child_manifest, _child_digest = registry_get_json(mirror, f"{repo}/manifests/{manifest_digest}")
            config = child_manifest.get("config", {})
            if config.get("digest"):
                blob_digests.add(config["digest"])
            for layer in child_manifest.get("layers", []):
                if layer.get("digest"):
                    blob_digests.add(layer["digest"])

        for blob_digest in sorted(blob_digests):
            logs.append(registry_get_blob(mirror, repo, blob_digest, dry_run=False))
        return image, True, "\n".join(logs)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        return image, False, str(e)


def pull_image(command: str, mirror_ref: str, original: str, dry_run: bool, tls_verify: bool) -> tuple[str, bool, str]:
    cmd = [command, "pull"]
    if Path(command).name in {"podman", "buildah", "skopeo"}:
        cmd.append(f"--tls-verify={str(tls_verify).lower()}")
    cmd.append(mirror_ref)
    if dry_run:
        return original, True, " ".join(cmd)

    proc = subprocess.run(cmd, text=True, capture_output=True)
    output = (proc.stdout + proc.stderr).strip()
    return original, proc.returncode == 0, output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=None,
        help="HF dataset repo in Tmax/SWERL task-data format. Defaults to task_data_hf_repo from --tool-configs.",
    )
    parser.add_argument("--revision", default=None, help="Optional HF dataset revision.")
    parser.add_argument("--split", default="train", help="Dataset split to scan for env_config images. Defaults to train.")
    parser.add_argument(
        "--skip-env-config",
        action="store_true",
        help="Only inspect task-data image.txt files and explicitly supplied images.",
    )
    parser.add_argument(
        "--tool-configs",
        default=None,
        help=(
            "SWERL tool/env config JSON, or path to JSON, e.g. "
            '\'{"task_data_hf_repo":"hamishivi/swerl-tmax-15k","image":"python:3.12-slim"}\'. '
            "The image value is included in the warmed images."
        ),
    )
    parser.add_argument("--image", action="append", default=[], help="Additional image to warm. Repeat as needed.")
    parser.add_argument("--mirror", required=True, help="Registry mirror host[:port], e.g. node123:5000.")
    parser.add_argument(
        "--method",
        choices=("registry-api", "pull"),
        default="registry-api",
        help="Warm via registry HTTP API or local container CLI pull. Defaults to registry-api.",
    )
    parser.add_argument(
        "--pull-command",
        default="docker",
        help="Container CLI to use when --method=pull. Defaults to docker.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent pulls. Defaults to 8.")
    parser.add_argument(
        "--platform",
        default="linux/amd64",
        help="Platform to warm from multi-arch indexes, or 'all'. Defaults to linux/amd64.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print pulls without running them.")
    parser.add_argument("--verbose", action="store_true", help="Print per-image success output.")
    parser.add_argument("--images-out", default=None, help="Optional path to write discovered original image refs.")
    parser.add_argument(
        "--tls-verify",
        action="store_true",
        help="Use TLS verification for Podman/Buildah/Skopeo pulls. Default is false for the HTTP Beaker mirror.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tool_configs = load_tool_configs(args.tool_configs)
    dataset = args.dataset or tool_configs.get("task_data_hf_repo")
    if not dataset:
        raise ValueError("Provide --dataset or --tool-configs with task_data_hf_repo")

    task_data_dir = resolve_task_data_dir(dataset, revision=args.revision)
    images = set(discover_images(task_data_dir))
    if not args.skip_env_config:
        images.update(discover_env_config_images(dataset, split=args.split, revision=args.revision))
    config_image = tool_configs.get("image")
    if config_image:
        images.add(config_image)
    images.update(args.image)
    images = sorted(images)
    if args.images_out:
        Path(args.images_out).write_text("\n".join(images) + ("\n" if images else ""), encoding="utf-8")

    pulls: list[tuple[str, str]] = []
    skipped: list[str] = []
    for image in images:
        mirror_ref = mirror_ref_for_dockerhub_image(image, args.mirror)
        if mirror_ref is None:
            skipped.append(image)
            continue
        pulls.append((image, mirror_ref))

    print(f"Discovered {len(images)} unique images in {task_data_dir}")
    action = "Warming" if args.method == "registry-api" else "Pulling"
    print(f"{action} {len(pulls)} Docker Hub images through mirror {args.mirror}")
    if skipped:
        print(f"Skipping {len(skipped)} non-Docker-Hub images:")
        for image in skipped:
            print(f"  {image}")

    failures: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        if args.method == "registry-api":
            future_to_image = {
                executor.submit(warm_with_registry_api, image, args.mirror, args.dry_run, args.platform): image
                for image, _mirror_ref in pulls
            }
        else:
            future_to_image = {
                executor.submit(pull_image, args.pull_command, mirror_ref, image, args.dry_run, args.tls_verify): image
                for image, mirror_ref in pulls
            }
        completed = concurrent.futures.as_completed(future_to_image)
        for future in tqdm(completed, total=len(future_to_image), desc=action, unit="image"):
            image, ok, output = future.result()
            status = "OK" if ok else "FAILED"
            if args.verbose or not ok:
                tqdm.write(f"[{status}] {image}")
                if output:
                    tqdm.write(output)
            if not ok:
                failures.append((image, output))

    if failures:
        raise SystemExit(f"{len(failures)} image pulls failed")


if __name__ == "__main__":
    main()
