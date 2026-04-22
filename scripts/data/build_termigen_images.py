#!/usr/bin/env python3
"""Build and push per-task Docker images for agent-task-termigen.

Reads task-data from the HF dataset repo (download + extract
``task-data.tar.gz``), which contains one directory per task with the
Harbor 2.0 layout: ``instruction.md``, ``tests/``, ``image.txt`` (target
image tag), and ``environment/`` (build context including the
``Dockerfile``). For each unique content hash, builds the image from the
``environment/`` directory and pushes it to the configured DockerHub
repo. The image tag in ``image.txt`` was pre-computed at dataset
creation time from the same content hash, so nothing in the dataset
needs to be rewritten after the build.

Usage:
    python scripts/data/build_termigen_images.py
    python scripts/data/build_termigen_images.py --dry-run
    python scripts/data/build_termigen_images.py --use-buildx --platform linux/amd64
    python scripts/data/build_termigen_images.py --cache-registry hamishivi/termigen-buildcache
"""

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request

from huggingface_hub import snapshot_download
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)


def _ensure_buildx_builder(builder_name: str) -> None:
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
        subprocess.run(["docker", "buildx", "use", builder_name], check=True)
    subprocess.run(["docker", "buildx", "inspect", "--bootstrap"], check=True)


def list_dockerhub_tags(namespace_repo: str) -> set:
    """Return every tag of ``<namespace>/<repo>`` from DockerHub in one paginated
    call sequence (anonymous; works for public repos).

    Falls back to an empty set on error; the caller should then rely on
    per-tag ``docker manifest inspect`` checks instead.
    """
    tags: set = set()
    base = f"https://hub.docker.com/v2/repositories/{namespace_repo}/tags"
    url = f"{base}?page_size=100"
    while url:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning(f"DockerHub repo {namespace_repo} not found (404); assuming empty.")
                return tags
            logger.warning(f"DockerHub tag listing failed for {namespace_repo}: {e}")
            return set()
        except Exception as e:
            logger.warning(f"DockerHub tag listing failed for {namespace_repo}: {e}")
            return set()
        for result in data.get("results", []):
            name = result.get("name")
            if name:
                tags.add(name)
        url = data.get("next")
    logger.info(f"Found {len(tags)} existing tag(s) under {namespace_repo}")
    return tags


def resolve_task_data_dir(repo_id: str) -> str:
    """Download + extract ``task-data.tar.gz`` from the dataset repo."""
    repo_dir = snapshot_download(repo_id, repo_type="dataset")
    tarball = os.path.join(repo_dir, "task-data.tar.gz")
    if not os.path.isfile(tarball):
        raise FileNotFoundError(f"No task-data.tar.gz in {repo_dir}")
    extract_dir = tarball + ".extracted"
    if not os.path.isdir(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        logger.info(f"Extracting {tarball} to {extract_dir}...")
        subprocess.run(["tar", "-xzf", tarball, "-C", extract_dir], check=True)
    return extract_dir


def main():
    parser = argparse.ArgumentParser(
        description="Build per-task Docker images for agent-task-termigen"
    )
    parser.add_argument(
        "--dataset",
        default="hamishivi/agent-task-termigen",
        help="HF dataset repo containing task-data.tar.gz with per-task image.txt.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-tasks", type=int, default=None, help="Limit on number of unique images built."
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--platform",
        default="linux/amd64",
        help="Docker platform (for example linux/amd64 or linux/amd64,linux/arm64).",
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
        default="termigen-builder",
        help="Buildx builder name to use when --use-buildx is enabled.",
    )
    parser.add_argument(
        "--quiet-build",
        action="store_true",
        help="Suppress streaming docker build output.",
    )
    parser.add_argument(
        "--cache-registry",
        default=None,
        help=(
            "Optional registry ref to use as a BuildKit shared cache "
            "(e.g. 'hamishivi/termigen-buildcache'). Implies --use-buildx."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stream_build_logs = not args.quiet_build

    logger.info(f"Resolving task data for {args.dataset}...")
    task_data_dir = resolve_task_data_dir(args.dataset)
    logger.info(f"Task data at {task_data_dir}")

    # Map image_tag -> (build_context_dir, [task_ids])
    tag_to_context: dict[str, str] = {}
    tag_to_tasks: dict[str, list[str]] = {}
    skipped_no_image = 0
    skipped_no_dockerfile = 0
    for tid in sorted(os.listdir(task_data_dir)):
        task_dir = os.path.join(task_data_dir, tid)
        if not os.path.isdir(task_dir):
            continue
        image_txt = os.path.join(task_dir, "image.txt")
        env_dir = os.path.join(task_dir, "environment")
        dockerfile = os.path.join(env_dir, "Dockerfile")
        if not os.path.isfile(image_txt):
            skipped_no_image += 1
            continue
        if not os.path.isfile(dockerfile):
            skipped_no_dockerfile += 1
            continue
        with open(image_txt, encoding="utf-8") as f:
            tag = f.read().strip()
        if not tag:
            skipped_no_image += 1
            continue
        tag_to_context.setdefault(tag, env_dir)
        tag_to_tasks.setdefault(tag, []).append(tid)

    logger.info(
        f"{len(tag_to_context)} unique images across "
        f"{sum(len(v) for v in tag_to_tasks.values())} tasks "
        f"(skipped {skipped_no_image} missing image.txt, "
        f"{skipped_no_dockerfile} missing Dockerfile)."
    )

    items = list(tag_to_context.items())

    existing_tags: set = set()
    if not args.force_rebuild:
        repos = {tag.split(":", 1)[0] for tag, _ in items}
        for repo in sorted(repos):
            existing_tags |= {f"{repo}:{t}" for t in list_dockerhub_tags(repo)}
        if existing_tags:
            before = len(items)
            items = [(tag, ctx) for tag, ctx in items if tag not in existing_tags]
            logger.info(
                f"Skipping {before - len(items)} image(s) already on DockerHub; "
                f"{len(items)} remaining to build."
            )

    if args.max_tasks is not None:
        items = items[: args.max_tasks]
        logger.info(f"Limiting to first {len(items)} unique images due to --max-tasks.")

    use_buildx = args.use_buildx or "," in args.platform or bool(args.cache_registry)
    multi_platform = "," in args.platform
    if use_buildx and not args.dry_run:
        _ensure_buildx_builder(args.buildx_builder)
    effective_workers = args.workers
    if multi_platform and args.workers != 1:
        logger.warning(
            "Multi-platform buildx is forced to workers=1 to avoid shared builder races "
            "(requested=%s).",
            args.workers,
        )
        effective_workers = 1

    lock = threading.Lock()
    counters = {"pushed": 0, "skipped": 0, "failed": 0}
    failed_tags: list[str] = []

    def build_one(image_tag: str, build_ctx: str) -> str:
        """Return one of 'pushed' | 'skipped' | 'failed'.

        Existence on DockerHub is already filtered upstream via the tag
        listing, so anything reaching here needs a build.
        """
        logger.info(f"Building {image_tag} from {build_ctx}...")
        try:
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
                cmd.append(build_ctx)
                subprocess.run(cmd, check=True)
            else:
                build_cmd = ["docker", "build", "--platform", args.platform, "-t", image_tag]
                if not stream_build_logs:
                    build_cmd.append("--quiet")
                build_cmd.append(build_ctx)
                push_cmd = ["docker", "push", image_tag]
                if not stream_build_logs:
                    push_cmd.insert(2, "--quiet")
                stdout = subprocess.DEVNULL if not stream_build_logs else None
                subprocess.run(build_cmd, check=True, stdout=stdout)
                subprocess.run(push_cmd, check=True, stdout=stdout)
                subprocess.run(["docker", "rmi", image_tag], check=False, stdout=stdout, stderr=stdout)
            logger.info(f"Pushed {image_tag}")
            return "pushed"
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build {image_tag}: {e}")
            with lock:
                failed_tags.append(image_tag)
            return "failed"

    if args.dry_run:
        for tag, ctx in items:
            logger.info(f"[DRY RUN] Would build {tag} from {ctx} ({len(tag_to_tasks[tag])} tasks)")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(build_one, tag, ctx): tag for tag, ctx in items}
        with logging_redirect_tqdm():
            progress = tqdm(total=len(futures), desc="Images", unit="img", smoothing=0.1)
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
    if failed_tags:
        logger.warning(f"{len(failed_tags)} image(s) failed. First 10: {failed_tags[:10]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
