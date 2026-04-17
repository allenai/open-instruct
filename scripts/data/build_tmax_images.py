#!/usr/bin/env python3
"""Build and push per-task Docker images for tmax-tasks dataset.

Converts Singularity container_def to Dockerfile, builds, and pushes.
Deduplicates by content hash to avoid rebuilding identical images.

Usage:
    python scripts/data/build_tmax_images.py --registry hamishivi
    python scripts/data/build_tmax_images.py --registry hamishivi --dry-run
"""

import argparse
import concurrent.futures
import hashlib
import io
import logging
import os
import subprocess
import tempfile
import threading

import docker as docker_sdk
from datasets import load_dataset

logger = logging.getLogger(__name__)


def container_def_to_dockerfile(container_def: str) -> str:
    """Convert a Singularity/Apptainer def to a Dockerfile."""
    base_image = "ubuntu:22.04"
    post_commands = []

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

    post_script = "\n".join(post_commands).strip()

    # Write post commands as a script and execute it, since multi-line RUN
    # with raw shell commands breaks Dockerfile parsing.
    return (
        f"FROM {base_image}\n"
        f"ENV DEBIAN_FRONTEND=noninteractive\n"
        f"COPY setup.sh /tmp/setup.sh\n"
        f"RUN chmod +x /tmp/setup.sh && /tmp/setup.sh\n"
        f"WORKDIR /workspace\n"
    ), post_script


def main():
    parser = argparse.ArgumentParser(description="Build per-task Docker images for tmax-tasks")
    parser.add_argument("--input", default="osieosie/tmax-tasks-skill-taxonomy-20260324-1k-verified")
    parser.add_argument("--registry", default="hamishivi", help="DockerHub username/org")
    parser.add_argument("--repo-prefix", default="swerl-tmax", help="Image repo prefix")
    parser.add_argument("--output-dataset", default="hamishivi/swerl-tmax-skill-taxonomy-1k")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8, help="Parallel build+push workers")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ds = load_dataset(args.input, split="train")
    logger.info(f"Loaded {len(ds)} tasks")

    if args.max_tasks:
        ds = ds.select(range(min(args.max_tasks, len(ds))))

    client = None if args.dry_run else docker_sdk.from_env()

    # Deduplicate by container_def hash
    hash_to_tasks: dict[str, list[str]] = {}
    hash_to_dockerfile: dict[str, str] = {}
    task_to_image: dict[str, str] = {}

    for row in ds:
        task_id = row["task_id"]
        dockerfile_content, setup_script = container_def_to_dockerfile(row["container_def"])
        combined = dockerfile_content + setup_script
        content_hash = hashlib.sha256(combined.encode()).hexdigest()[:12]

        hash_to_tasks.setdefault(content_hash, []).append(task_id)
        if content_hash not in hash_to_dockerfile:
            hash_to_dockerfile[content_hash] = (dockerfile_content, setup_script)

    logger.info(f"{len(hash_to_dockerfile)} unique images for {len(ds)} tasks")

    # Build and push images (parallel)
    lock = threading.Lock()

    def build_one(content_hash: str, dockerfile_content: str, setup_script: str) -> None:
        image_tag = f"{args.registry}/{args.repo_prefix}:{content_hash}"
        tasks = hash_to_tasks[content_hash]

        # Skip if image already exists on DockerHub
        try:
            check = subprocess.run(
                ["docker", "manifest", "inspect", image_tag],
                capture_output=True, timeout=15,
            )
            if check.returncode == 0:
                logger.info(f"Skipping {image_tag} (exists on registry)")
                with lock:
                    for t in tasks:
                        task_to_image[t] = image_tag
                return
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

                image, build_logs = worker_client.images.build(
                    path=tmpdir,
                    tag=image_tag,
                    rm=True,
                )
            logger.info(f"Built {image_tag}")

            worker_client.images.push(args.registry + "/" + args.repo_prefix, tag=content_hash)
            logger.info(f"Pushed {image_tag}")

            # Clean up local image to save disk
            try:
                worker_client.images.remove(image_tag, force=True)
            except Exception:
                pass

            with lock:
                for t in tasks:
                    task_to_image[t] = image_tag

        except Exception as e:
            logger.error(f"Failed to build {image_tag}: {e}")
            with lock:
                for t in tasks:
                    task_to_image[t] = "ubuntu:22.04"

    if args.dry_run:
        for content_hash in hash_to_dockerfile:
            image_tag = f"{args.registry}/{args.repo_prefix}:{content_hash}"
            tasks = hash_to_tasks[content_hash]
            logger.info(f"[DRY RUN] Would build {image_tag} for {len(tasks)} tasks")
            for t in tasks:
                task_to_image[t] = image_tag
    else:
        items = list(hash_to_dockerfile.items())
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(build_one, ch, df, ss)
                for ch, (df, ss) in items
            ]
            for f in concurrent.futures.as_completed(futures):
                f.result()  # raise exceptions

    # Update the HF dataset with image tags
    logger.info(f"Updating dataset {args.output_dataset} with image tags...")
    out_ds = load_dataset(args.output_dataset, split="train")

    def update_image(example):
        tid = example["env_config"]["task_id"]
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
