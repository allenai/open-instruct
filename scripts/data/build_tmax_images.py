#!/usr/bin/env python3
"""Build and push per-task Docker images for tmax-tasks dataset.

Converts Singularity container_def to Dockerfile, builds, and pushes.
Deduplicates by content hash to avoid rebuilding identical images.

Usage:
    python scripts/data/build_tmax_images.py --registry hamishivi
    python scripts/data/build_tmax_images.py --registry hamishivi --dry-run
    python scripts/data/build_tmax_images.py --registry hamishivi --platform linux/amd64
    python scripts/data/build_tmax_images.py --registry hamishivi --platform linux/amd64,linux/arm64 --use-buildx
"""

import argparse
import concurrent.futures
import hashlib
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading

import docker as docker_sdk
from datasets import load_dataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)


_BUILT_IMAGE_ID_RE = re.compile(r"(?:Successfully built |sha256:)([0-9a-f]+)")


def _format_build_chunk_line(chunk: dict) -> str | None:
    """Best-effort human line from a Docker engine JSON build message."""
    stream = chunk.get("stream")
    if stream is not None:
        if isinstance(stream, bytes):
            stream = stream.decode("utf-8", "replace")
        line = str(stream).rstrip()
        if line:
            return line
    status = chunk.get("status")
    if not status:
        return None
    parts = [str(status)]
    if chunk.get("id"):
        parts.append(str(chunk["id"]))
    prog = chunk.get("progressDetail") or {}
    if isinstance(prog, dict) and prog.get("current") is not None and prog.get("total"):
        parts.append(f"{prog['current']}/{prog['total']}")
    return " ".join(parts)


def _stream_classic_build(worker_client, tmpdir, *, image_tag, platform, quiet):
    """Build using the low-level API so logs stream in real time.

    docker.DockerClient.images.build() fully consumes the build stream
    before returning (to locate the image id), so you see no output
    until the build finishes. Using api.build(decode=True) yields JSON
    chunks as the daemon emits them.
    """
    api = worker_client.api
    image_id: str | None = None
    for chunk in api.build(
        path=tmpdir, tag=image_tag, rm=True, platform=platform, decode=True
    ):
        if not isinstance(chunk, dict):
            continue
        err = chunk.get("errorDetail") or chunk.get("error")
        if err:
            msg = err if isinstance(err, str) else err.get("message", str(err))
            raise RuntimeError(f"Docker build error for {image_tag}: {msg}")
        stream = chunk.get("stream")
        if stream:
            if isinstance(stream, bytes):
                stream = stream.decode("utf-8", "replace")
            match = _BUILT_IMAGE_ID_RE.search(stream)
            if match:
                image_id = match.group(1)
        if quiet:
            continue
        line = _format_build_chunk_line(chunk)
        if line:
            sys.stdout.write(f"{image_tag} | {line}\n")
            sys.stdout.flush()
    if image_id is None:
        raise RuntimeError(f"Docker build for {image_tag} did not report an image id")
    return worker_client.images.get(image_id)


def _ensure_buildx_builder(builder_name: str) -> None:
    """Ensure a docker-container buildx builder exists and is selected."""
    inspect = subprocess.run(
        ["docker", "buildx", "inspect", builder_name],
        capture_output=True,
        text=True,
    )
    if inspect.returncode != 0:
        logger.info(f"Creating buildx builder '{builder_name}' with docker-container driver")
        subprocess.run(
            ["docker", "buildx", "create", "--name", builder_name, "--driver", "docker-container", "--use"],
            check=True,
        )
    else:
        # Force using the intended builder in case another one is currently active.
        subprocess.run(["docker", "buildx", "use", builder_name], check=True)

    # Bootstrap so QEMU/binfmt capabilities are initialized if available.
    subprocess.run(["docker", "buildx", "inspect", "--bootstrap"], check=True)


# Matches simple apt installs that we can safely hoist into a shared layer.
# Allows an optional leading `apt-get update &&` and known benign flags.
_APT_INSTALL_RE = re.compile(
    r"^\s*(?:apt-get\s+update\s*&&\s*)?"
    r"apt-get\s+install\s+-y(?:\s+--no-install-recommends)?\s+"
    r"(?P<pkgs>[A-Za-z0-9][A-Za-z0-9._+\-\s=:]*?)"
    r"\s*(?:&&\s*rm\s+-rf\s+/var/lib/apt/lists/\*)?\s*$"
)

# Matches simple pip installs: `pip install ...` or `pip3 install ...`, packages only.
# Any flag (--index-url, --extra-index-url, -e, --no-cache-dir, etc.) is left in the
# residual script so we don't mishandle non-trivial installs.
_PIP_INSTALL_RE = re.compile(
    r"^\s*(?P<exe>pip3?)\s+install\s+(?:--no-cache-dir\s+)?"
    r"(?P<pkgs>[A-Za-z0-9][A-Za-z0-9._+\-\s\[\]=<>!~]*?)\s*$"
)


def _is_valid_pip_package(token: str) -> bool:
    """Reject anything that looks like a flag, URL, or VCS ref."""
    if not token or token.startswith("-"):
        return False
    if "://" in token or token.startswith(("git+", "http", "https", "file:", "/")):
        return False
    return True


def split_package_installs(post_script: str) -> tuple[list[str], list[str], str]:
    """Extract simple apt/pip install commands from a Singularity %post script.

    Returns:
        (apt_packages_sorted_unique, pip_packages_sorted_unique, residual_script)

    Lines that look like ``apt-get install -y a b c`` or ``pip install x y`` are
    removed from the returned residual. Anything with unusual flags, URLs, or
    version pinning that is not purely ``pkg==x.y`` syntax stays in residual so
    that its semantics is preserved verbatim.
    """
    apt: list[str] = []
    pip: list[str] = []
    residual_lines: list[str] = []
    for line in post_script.splitlines():
        m_apt = _APT_INSTALL_RE.match(line)
        if m_apt:
            apt.extend(tok for tok in m_apt.group("pkgs").split() if tok and not tok.startswith("-"))
            continue
        m_pip = _PIP_INSTALL_RE.match(line)
        if m_pip:
            pkgs = m_pip.group("pkgs").split()
            if pkgs and all(_is_valid_pip_package(p) for p in pkgs):
                pip.extend(pkgs)
                continue
        residual_lines.append(line)
    apt = sorted(set(apt))
    pip = sorted(set(pip))
    residual = "\n".join(residual_lines).strip()
    return apt, pip, residual


def _parse_post(container_def: str) -> tuple[str, str]:
    """Return (base_image, post_script_text) from a Singularity def file."""
    base_image = "ubuntu:22.04"
    post_commands: list[str] = []

    for line in container_def.splitlines():
        stripped = line.strip()
        if stripped.startswith("From:"):
            base_image = stripped.split(":", 1)[1].strip()

    in_post = False
    for line in container_def.splitlines():
        if line.strip() == "%post":
            in_post = True
            continue
        if line.strip().startswith("%") and in_post:
            break
        if in_post:
            post_commands.append(line)

    return base_image, "\n".join(post_commands).strip()


# Verifier dependency baked into every task image. Some multi_protocol verifiers
# import `requests`; it was provided by base_intricate.sif at solve time but is
# not part of the per-task container.def, so the published prebuilt image must
# add it. Emitted as the final layer, after the task %post has installed pip, so
# it is a fast no-op for tasks that already pull it in.
_BAKE_REQUESTS = "RUN pip3 install --no-cache-dir requests"


def _parse_task_id_values(values: list[str] | None) -> set[str]:
    """Parse repeated comma/whitespace-separated task-id arguments."""
    task_ids: set[str] = set()
    for value in values or []:
        for token in re.split(r"[\s,]+", value.strip()):
            if token:
                task_ids.add(token)
    return task_ids


def container_def_to_dockerfile(
    container_def: str, *, split_pkg_layers: bool = False
) -> tuple[str, str]:
    """Convert a Singularity/Apptainer def to a Dockerfile + setup.sh pair.

    When ``split_pkg_layers`` is True, simple apt and pip install commands are
    hoisted into their own RUN layers (sorted + deduped) so Docker Hub can
    share them across tasks with the same install sets. The remaining `%post`
    body is written into setup.sh and executed in a separate layer.
    """
    base_image, post_script = _parse_post(container_def)

    if not split_pkg_layers:
        return (
            f"FROM {base_image}\n"
            f"ENV DEBIAN_FRONTEND=noninteractive\n"
            f"COPY setup.sh /tmp/setup.sh\n"
            f"RUN bash /tmp/setup.sh && rm -f /tmp/setup.sh\n"
            f"{_BAKE_REQUESTS}\n"
        ), post_script

    apt_pkgs, pip_pkgs, residual = split_package_installs(post_script)

    # If the residual still contains any apt/apt-get reference (e.g. multi-line
    # apt-get install with `\` continuation that the simple regex doesn't
    # hoist), make sure apt lists are refreshed in the residual — the hoisted
    # apt layer rm's /var/lib/apt/lists/* to keep its layer small.
    if residual and apt_pkgs and re.search(r"\bapt(?:-get)?\b", residual):
        residual = "apt-get update\n" + residual

    lines = [
        f"FROM {base_image}",
        "ENV DEBIAN_FRONTEND=noninteractive",
    ]
    if apt_pkgs:
        pkg_str = " ".join(apt_pkgs)
        lines.append(
            "RUN apt-get update "
            "&& apt-get install -y --no-install-recommends "
            f"{pkg_str} "
            "&& rm -rf /var/lib/apt/lists/*"
        )
    if pip_pkgs:
        pkg_str = " ".join(pip_pkgs)
        lines.append(f"RUN pip3 install --no-cache-dir {pkg_str}")
    if residual:
        lines.append("COPY setup.sh /tmp/setup.sh")
        lines.append("RUN bash /tmp/setup.sh && rm -f /tmp/setup.sh")
    lines.append(_BAKE_REQUESTS)
    dockerfile = "\n".join(lines) + "\n"
    return dockerfile, residual


def main():
    parser = argparse.ArgumentParser(description="Build per-task Docker images for tmax-tasks")
    parser.add_argument("--input", default="osieosie/tmax-tasks-skill-taxonomy-20260324-1k-verified")
    parser.add_argument("--registry", default="hamishivi", help="DockerHub username/org")
    parser.add_argument("--repo-prefix", default="swerl-tmax", help="Image repo prefix")
    parser.add_argument("--output-dataset", default="hamishivi/swerl-tmax-skill-taxonomy-1k")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8, help="Parallel build+push workers")
    parser.add_argument(
        "--platform",
        default="linux/amd64",
        help="Docker platform to build for (for example linux/amd64 or linux/arm64).",
    )
    parser.add_argument(
        "--use-buildx",
        action="store_true",
        help="Build and push with docker buildx (required for multi-platform tags).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild and repush tags even if they already exist on the registry.",
    )
    parser.add_argument(
        "--buildx-builder",
        default="swerl-multiarch",
        help="Buildx builder name to use when --use-buildx is enabled.",
    )
    parser.add_argument(
        "--quiet-build",
        action="store_true",
        help="Suppress streaming docker build output (default: print build logs).",
    )
    parser.add_argument(
        "--verbose-build",
        action="store_true",
        help="Alias for default behavior (logs on); kept for backward compatibility.",
    )
    parser.add_argument(
        "--repair-missing",
        action="store_true",
        help=(
            "Rebuild only tasks whose image in the output dataset is currently "
            "a fallback (ubuntu:22.04) from a prior failed build. Leaves other "
            "tasks untouched."
        ),
    )
    parser.add_argument(
        "--split-pkg-layers",
        action="store_true",
        help=(
            "Hoist simple apt-get / pip install commands into their own RUN "
            "layers so Docker Hub can dedupe them across tasks. Changes the "
            "per-task content hash (expect a one-time full rebuild)."
        ),
    )
    parser.add_argument(
        "--cache-registry",
        default=None,
        help=(
            "Optional registry ref to use as a BuildKit shared cache (e.g. "
            "'hamishi740/swerl-tmax-buildcache'). Enables "
            "--cache-to type=registry,mode=max,ref=<ref> and matching "
            "--cache-from, so repeated RUN instructions (like shared apt/pip "
            "layers) are reused across builds without re-executing them. "
            "Implies --use-buildx."
        ),
    )
    parser.add_argument(
        "--task-ids",
        action="append",
        default=None,
        help=(
            "Optional comma/whitespace-separated task ids to rebuild and update. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--task-ids-file",
        default=None,
        help="Optional file containing task ids to rebuild, separated by commas or whitespace.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stream_build_logs = not args.quiet_build
    if args.verbose_build and args.quiet_build:
        logger.warning("--verbose-build overrides --quiet-build: build logs will be printed.")
        stream_build_logs = True

    ds = load_dataset(args.input, split="train")
    logger.info(f"Loaded {len(ds)} tasks")

    if args.max_tasks:
        ds = ds.select(range(min(args.max_tasks, len(ds))))

    selected_task_ids = _parse_task_id_values(args.task_ids)
    if args.task_ids_file:
        with open(args.task_ids_file) as f:
            selected_task_ids.update(_parse_task_id_values([f.read()]))
    if selected_task_ids:
        logger.info(f"--task-ids: selected {len(selected_task_ids)} task(s).")

    target_task_ids: set[str] | None = set(selected_task_ids) if selected_task_ids else None
    out_ds_probe = None
    if args.repair_missing:
        if args.dry_run:
            logger.info("--repair-missing is a no-op under --dry-run (dataset is not written).")
        logger.info(
            f"--repair-missing: scanning {args.output_dataset} for tasks with image=='ubuntu:22.04'..."
        )
        out_ds_probe = load_dataset(args.output_dataset, split="train")
        fallback_task_ids = set()
        for row in out_ds_probe:
            env_config = row.get("env_config") or {}
            image = env_config.get("image")
            tid = env_config.get("task_id")
            if image == "ubuntu:22.04" and tid:
                fallback_task_ids.add(tid)
        logger.info(
            f"--repair-missing: {len(fallback_task_ids)} task(s) in {args.output_dataset} "
            "are fallbacks and will be rebuilt."
        )
        target_task_ids = (
            fallback_task_ids if target_task_ids is None else target_task_ids.intersection(fallback_task_ids)
        )
        if not target_task_ids:
            logger.info("Nothing to repair. Exiting.")
            return

    if target_task_ids is not None:
        if out_ds_probe is None:
            out_ds_probe = load_dataset(args.output_dataset, split="train")
        output_task_ids = {
            row.get("env_config", {}).get("task_id")
            for row in out_ds_probe
            if isinstance(row.get("env_config"), dict)
        }
        missing_output_ids = target_task_ids - output_task_ids
        if missing_output_ids:
            raise KeyError(
                f"{args.output_dataset} is missing {len(missing_output_ids)} selected task id(s): "
                f"{sorted(missing_output_ids)}"
            )

    client = None if args.dry_run else docker_sdk.from_env()

    # Deduplicate by container_def hash
    hash_to_tasks: dict[str, list[str]] = {}
    hash_to_dockerfile: dict[str, str] = {}
    task_to_image: dict[str, str] = {}
    seen_task_ids: set[str] = set()

    for row in ds:
        env_config = row.get("env_config") if isinstance(row, dict) else None
        task_id = row.get("task_id") or row.get("ground_truth")
        if not task_id and isinstance(env_config, dict):
            task_id = env_config.get("task_id")
        if not task_id:
            raise KeyError("Missing task identifier: expected task_id, ground_truth, or env_config.task_id")

        seen_task_ids.add(task_id)
        if target_task_ids is not None and task_id not in target_task_ids:
            continue

        container_def = row.get("container_def")
        if not container_def:
            raise KeyError(
                f"Missing container_def for task {task_id}. "
                "Use a source dataset that includes container_def (for example osieosie/tmax-tasks-*)."
            )
        dockerfile_content, setup_script = container_def_to_dockerfile(
            container_def, split_pkg_layers=args.split_pkg_layers
        )
        combined = dockerfile_content + setup_script
        content_hash = hashlib.sha256(combined.encode()).hexdigest()[:12]

        hash_to_tasks.setdefault(content_hash, []).append(task_id)
        if content_hash not in hash_to_dockerfile:
            hash_to_dockerfile[content_hash] = (dockerfile_content, setup_script)

    if target_task_ids is not None:
        missing_source_ids = target_task_ids - seen_task_ids
        if missing_source_ids:
            raise KeyError(
                f"{args.input} is missing {len(missing_source_ids)} selected task id(s): "
                f"{sorted(missing_source_ids)}"
            )
        logger.info(
            f"{len(hash_to_dockerfile)} unique images to rebuild for {len(target_task_ids)} selected task(s)."
        )
    else:
        logger.info(f"{len(hash_to_dockerfile)} unique images for {len(ds)} tasks")

    # Build and push images (parallel)
    lock = threading.Lock()
    use_buildx = args.use_buildx or "," in args.platform or bool(args.cache_registry)
    multi_platform = "," in args.platform
    if args.cache_registry:
        logger.info(
            "BuildKit registry cache enabled: ref=%s (cache-to mode=max, cache-from).",
            args.cache_registry,
        )
    if use_buildx and not args.dry_run:
        _ensure_buildx_builder(args.buildx_builder)
    effective_workers = args.workers
    if multi_platform and args.workers != 1:
        # Multi-arch buildx builds thrash the shared docker-container builder when
        # multiple workers drive it in parallel. Keep that path serialized.
        logger.warning(
            "Multi-platform buildx is forced to workers=1 to avoid shared builder "
            "lifecycle races (requested=%s).",
            args.workers,
        )
        effective_workers = 1

    counters = {"pushed": 0, "skipped": 0, "failed": 0}

    def build_one(content_hash: str, dockerfile_content: str, setup_script: str) -> str:
        """Return one of 'pushed' | 'skipped' | 'failed'."""
        image_tag = f"{args.registry}/{args.repo_prefix}:{content_hash}"
        tasks = hash_to_tasks[content_hash]

        # Skip if image already exists on DockerHub unless force-rebuild is requested.
        if not args.force_rebuild:
            try:
                check = subprocess.run(
                    ["docker", "manifest", "inspect", image_tag],
                    capture_output=True,
                    timeout=15,
                )
                if check.returncode == 0:
                    logger.info(f"Skipping {image_tag} (exists on registry)")
                    with lock:
                        for t in tasks:
                            task_to_image[t] = image_tag
                    return "skipped"
            except Exception:
                pass

        # Each worker needs its own Docker client
        worker_client = docker_sdk.from_env(timeout=300)

        logger.info(f"Building {image_tag}...")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
                    f.write(dockerfile_content)
                with open(os.path.join(tmpdir, "setup.sh"), "w") as f:
                    f.write(setup_script)

                if use_buildx:
                    cmd = [
                        "docker",
                        "buildx",
                        "build",
                        "--platform",
                        args.platform,
                        "--builder",
                        args.buildx_builder,
                        "--tag",
                        image_tag,
                        "--push",
                    ]
                    if args.cache_registry:
                        cmd.extend([
                            "--cache-to",
                            f"type=registry,ref={args.cache_registry},mode=max",
                            "--cache-from",
                            f"type=registry,ref={args.cache_registry}",
                        ])
                    if stream_build_logs:
                        cmd.extend(["--progress", "plain"])
                    cmd.append(tmpdir)
                    subprocess.run(cmd, check=True)
                    image = None
                else:
                    image = _stream_classic_build(
                        worker_client,
                        tmpdir,
                        image_tag=image_tag,
                        platform=args.platform,
                        quiet=not stream_build_logs,
                    )
            logger.info(f"Built {image_tag}")

            if not use_buildx:
                worker_client.images.push(args.registry + "/" + args.repo_prefix, tag=content_hash)
                logger.info(f"Pushed {image_tag}")
            else:
                logger.info(f"Pushed multi-platform image {image_tag}")

            # Clean up local image to save disk
            if not use_buildx:
                try:
                    worker_client.images.remove(image_tag, force=True)
                except Exception:
                    pass

            with lock:
                for t in tasks:
                    task_to_image[t] = image_tag
            return "pushed"

        except Exception as e:
            logger.error(f"Failed to build {image_tag}: {e}")
            with lock:
                for t in tasks:
                    task_to_image[t] = "ubuntu:22.04"
            return "failed"

    if args.dry_run:
        for content_hash in hash_to_dockerfile:
            image_tag = f"{args.registry}/{args.repo_prefix}:{content_hash}"
            tasks = hash_to_tasks[content_hash]
            logger.info(f"[DRY RUN] Would build {image_tag} for {len(tasks)} tasks")
            for t in tasks:
                task_to_image[t] = image_tag
    else:
        items = list(hash_to_dockerfile.items())
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(build_one, ch, df, ss)
                for ch, (df, ss) in items
            ]
            with logging_redirect_tqdm():
                progress = tqdm(
                    total=len(futures),
                    desc="Images",
                    unit="img",
                    smoothing=0.1,
                )
                try:
                    for f in concurrent.futures.as_completed(futures):
                        try:
                            outcome = f.result()
                        except Exception:
                            outcome = "failed"
                            logger.exception("build_one raised")
                        counters[outcome] = counters.get(outcome, 0) + 1
                        progress.set_postfix(
                            pushed=counters["pushed"],
                            skipped=counters["skipped"],
                            failed=counters["failed"],
                        )
                        progress.update(1)
                finally:
                    progress.close()
        logger.info(
            "Build run complete: pushed=%d skipped=%d failed=%d",
            counters["pushed"], counters["skipped"], counters["failed"],
        )

    # Update the HF dataset with image tags.
    # Under --repair-missing we only overwrite the tasks we rebuilt and keep the
    # existing image for every other row.
    logger.info(f"Updating dataset {args.output_dataset} with image tags...")
    out_ds = out_ds_probe if out_ds_probe is not None else load_dataset(args.output_dataset, split="train")
    allowed_overrides = target_task_ids

    def update_image(example):
        tid = example["env_config"]["task_id"]
        if allowed_overrides is not None and tid not in allowed_overrides:
            return example
        if tid in task_to_image:
            example["env_config"]["image"] = task_to_image[tid]
        return example

    out_ds = out_ds.map(update_image)

    if not args.dry_run:
        out_ds.push_to_hub(args.output_dataset, split="train")
        logger.info("Dataset updated!")
    else:
        sample = out_ds[0]["env_config"]
        logger.info(f"[DRY RUN] Sample env_config would be: {sample}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
