"""Create agent_task datasets from various source formats.

Supports three input formats:

1. **tar** — HF dataset with gzip-compressed tar archives (e.g., OpenThoughts-Agent-v1-RL).
   Each row has ``path`` (task ID) and ``task_binary`` (gzip-compressed tar bytes).

2. **directory** — Local directory where each subdirectory is a task with
   ``instruction.md``, ``environment/Dockerfile``, ``environment/seeds/``, ``tests/``, etc.

3. **swe_gym** — HF dataset with SWE-Bench/SWE-Gym columns: ``instance_id``,
   ``problem_statement``, ``repo``, ``base_commit``, ``test_patch``, ``FAIL_TO_PASS``.
   Uses pre-built Docker images; no Dockerfile building needed.

Usage examples:

    # From tar-based HF dataset
    python scripts/data/create_agent_task_dataset.py \\
        --input_dataset open-thoughts/OpenThoughts-Agent-v1-RL \\
        --input_format tar \\
        --output_dir data/agent_tasks \\
        --build_images

    # From directory-based repo
    python scripts/data/create_agent_task_dataset.py \\
        --input_dir /path/to/endless-terminals \\
        --input_format directory \\
        --output_dir data/agent_tasks \\
        --build_images

    # From SWE-Gym (pre-built images, no build needed)
    python scripts/data/create_agent_task_dataset.py \\
        --input_dataset SWE-Gym/SWE-Gym \\
        --input_format swe_gym \\
        --output_dir data/agent_tasks

    # Build + push to registry
    python scripts/data/create_agent_task_dataset.py \\
        --input_dir /path/to/data \\
        --input_format directory \\
        --output_dir data/agent_tasks \\
        --build_images \\
        --registry ghcr.io/myorg

    # Skip image building, use default image for all tasks
    python scripts/data/create_agent_task_dataset.py \\
        --input_dir /path/to/data \\
        --input_format directory \\
        --output_dir data/agent_tasks \\
        --default_image ubuntu:24.04
"""

import argparse
import gzip
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import tarfile

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful coding assistant. You have access to a bash terminal and a file editor. "
    "Use the tools to explore the codebase, understand the problem, implement a solution, and "
    "verify it works. When you are confident your solution is correct, use the submit tool."
)

# Anti-peeking setup script for SWE-Gym tasks
_SWE_GYM_SETUP_SH = r"""#!/bin/bash
# Sanitize git history to prevent the model from cheating via git log/reflog.
# See: https://github.com/SWE-bench/SWE-bench/issues/465
cd /workspace 2>/dev/null || exit 0
if [ -d .git ]; then
    git reflog expire --expire=now --all 2>/dev/null
    git gc --prune=now --aggressive 2>/dev/null
    # Remove reflog files directly as a fallback
    rm -rf .git/logs 2>/dev/null
fi
"""


def extract_tar_dataset(input_dataset: str, output_dir: str, split: str = "train") -> list[str]:
    """Extract tasks from a tar-based HF dataset.

    Args:
        input_dataset: HuggingFace dataset path (e.g., "open-thoughts/OpenThoughts-Agent-v1-RL").
        output_dir: Directory to write extracted task files.
        split: Dataset split to use.

    Returns:
        List of task IDs extracted.
    """
    from datasets import load_dataset

    logger.info(f"Loading dataset {input_dataset} (split={split})")
    ds = load_dataset(input_dataset, split=split)

    task_ids = []
    for row in ds:
        task_id = row["path"]
        task_dir = os.path.join(output_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)

        # Extract gzip-compressed tar
        tar_bytes = row["task_binary"]
        with gzip.open(io.BytesIO(tar_bytes), "rb") as gz:
            with tarfile.open(fileobj=gz, mode="r:") as tar:
                tar.extractall(path=task_dir, filter="data")

        task_ids.append(task_id)

    logger.info(f"Extracted {len(task_ids)} tasks from tar dataset")
    return task_ids


def extract_directory_dataset(input_dir: str, output_dir: str) -> list[str]:
    """Copy tasks from a directory-based source.

    Args:
        input_dir: Source directory where each subdirectory is a task.
        output_dir: Directory to write task files.

    Returns:
        List of task IDs copied.
    """
    task_ids = []
    for entry in sorted(os.listdir(input_dir)):
        src = os.path.join(input_dir, entry)
        if not os.path.isdir(src):
            continue
        # Must have at least instruction.md
        if not os.path.isfile(os.path.join(src, "instruction.md")):
            logger.debug(f"Skipping {entry}: no instruction.md")
            continue

        dst = os.path.join(output_dir, entry)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        task_ids.append(entry)

    logger.info(f"Copied {len(task_ids)} tasks from directory")
    return task_ids


def extract_swe_gym_dataset(input_dataset: str, output_dir: str, split: str = "train") -> list[str]:
    """Extract tasks from a SWE-Gym/SWE-Bench HF dataset.

    Uses pre-built Docker images from DockerHub. Generates instruction.md,
    image.txt, tests/test.sh, and setup.sh for anti-peeking.

    Args:
        input_dataset: HuggingFace dataset path (e.g., "SWE-Gym/SWE-Gym").
        output_dir: Directory to write task files.
        split: Dataset split to use.

    Returns:
        List of task IDs extracted.
    """
    from datasets import load_dataset

    logger.info(f"Loading SWE-Gym dataset {input_dataset} (split={split})")
    ds = load_dataset(input_dataset, split=split)

    task_ids = []
    for row in ds:
        instance_id = row["instance_id"]
        # Sanitize instance_id for filesystem (replace / and other special chars)
        task_id = instance_id.replace("/", "__")
        task_dir = os.path.join(output_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)

        # Write instruction
        problem_statement = row.get("problem_statement", "")
        repo = row.get("repo", "")
        base_commit = row.get("base_commit", "")
        instruction = (
            f"# {instance_id}\n\n"
            f"**Repository:** {repo}\n"
            f"**Base commit:** {base_commit}\n\n"
            f"## Problem Statement\n\n{problem_statement}\n\n"
            f"Please fix this issue in the repository at /workspace. "
            f"When you are done, use the submit tool to verify your fix."
        )
        with open(os.path.join(task_dir, "instruction.md"), "w") as f:
            f.write(instruction)

        # Write image.txt (pre-built SWE-bench images)
        image_tag = f"xingyaoww/sweb.eval.x86_64.{instance_id.replace('/', '__').lower()}"
        with open(os.path.join(task_dir, "image.txt"), "w") as f:
            f.write(image_tag)

        # Write test script
        test_patch = row.get("test_patch", "")
        fail_to_pass = row.get("FAIL_TO_PASS", "")
        if isinstance(fail_to_pass, list):
            fail_to_pass = " ".join(fail_to_pass)

        tests_dir = os.path.join(task_dir, "tests")
        os.makedirs(tests_dir, exist_ok=True)

        test_sh = _generate_swe_gym_test_sh(test_patch, fail_to_pass)
        with open(os.path.join(tests_dir, "test.sh"), "w") as f:
            f.write(test_sh)

        # Write anti-peeking setup.sh
        with open(os.path.join(task_dir, "setup.sh"), "w") as f:
            f.write(_SWE_GYM_SETUP_SH)

        task_ids.append(task_id)

    logger.info(f"Extracted {len(task_ids)} tasks from SWE-Gym dataset")
    return task_ids


def _generate_swe_gym_test_sh(test_patch: str, fail_to_pass: str) -> str:
    """Generate a test.sh script for SWE-Gym tasks."""
    # Escape the test patch for embedding in a heredoc
    return f"""#!/bin/bash
set -e

cd /workspace

# Apply test patch
cat << 'TEST_PATCH_EOF' | git apply --allow-empty -
{test_patch}
TEST_PATCH_EOF

# Run the failing tests
TESTS="{fail_to_pass}"
if [ -n "$TESTS" ]; then
    python -m pytest $TESTS -x --tb=short 2>&1
    EXIT_CODE=$?
else
    echo "No tests specified"
    EXIT_CODE=1
fi

# Write reward
mkdir -p /logs/verifier
if [ $EXIT_CODE -eq 0 ]; then
    echo "1" > /logs/verifier/reward.txt
else
    echo "0" > /logs/verifier/reward.txt
fi

exit $EXIT_CODE
"""


def build_and_push_images(
    output_dir: str,
    task_ids: list[str],
    registry: str | None = None,
) -> None:
    """Build Docker images from per-task Dockerfiles, deduplicating by content hash.

    Args:
        output_dir: Base directory containing task subdirectories.
        task_ids: List of task IDs to process.
        registry: Registry prefix for tagging (e.g., "ghcr.io/myorg"). If None, local only.
    """
    # Collect unique Dockerfiles by content hash
    hash_to_tasks: dict[str, list[str]] = {}
    hash_to_context: dict[str, str] = {}

    for task_id in task_ids:
        dockerfile_path = os.path.join(output_dir, task_id, "environment", "Dockerfile")
        if not os.path.isfile(dockerfile_path):
            logger.debug(f"Skipping {task_id}: no Dockerfile")
            continue

        with open(dockerfile_path, "rb") as f:
            content_hash = hashlib.sha256(f.read()).hexdigest()[:12]

        hash_to_tasks.setdefault(content_hash, []).append(task_id)
        # Use first task's environment dir as build context
        if content_hash not in hash_to_context:
            hash_to_context[content_hash] = os.path.join(output_dir, task_id, "environment")

    logger.info(f"Found {len(hash_to_context)} unique Dockerfiles for {len(task_ids)} tasks")

    # Build each unique image
    for content_hash, context_dir in hash_to_context.items():
        if registry:
            image_tag = f"{registry}/agent-task:{content_hash}"
        else:
            image_tag = f"agent-task:{content_hash}"

        logger.info(f"Building image {image_tag} from {context_dir}")
        subprocess.run(
            ["docker", "build", "-t", image_tag, context_dir],
            check=True,
        )

        if registry:
            logger.info(f"Pushing {image_tag}")
            subprocess.run(["docker", "push", image_tag], check=True)

        # Write image.txt for all tasks that share this Dockerfile
        for task_id in hash_to_tasks[content_hash]:
            image_file = os.path.join(output_dir, task_id, "image.txt")
            with open(image_file, "w") as f:
                f.write(image_tag)

    logger.info("Image build complete")


def write_default_images(output_dir: str, task_ids: list[str], default_image: str) -> None:
    """Write image.txt with a default image for all tasks that don't have one.

    Args:
        output_dir: Base directory containing task subdirectories.
        task_ids: List of task IDs to process.
        default_image: Default Docker image tag.
    """
    for task_id in task_ids:
        image_file = os.path.join(output_dir, task_id, "image.txt")
        if not os.path.isfile(image_file):
            with open(image_file, "w") as f:
                f.write(default_image)


def generate_jsonl(output_dir: str, task_ids: list[str], output_file: str) -> None:
    """Generate JSONL dataset file.

    Args:
        output_dir: Base directory containing task subdirectories.
        task_ids: List of task IDs.
        output_file: Path to output JSONL file.
    """
    with open(output_file, "w") as f:
        for task_id in task_ids:
            instruction_file = os.path.join(output_dir, task_id, "instruction.md")
            if os.path.isfile(instruction_file):
                with open(instruction_file) as inf:
                    instruction = inf.read().strip()
            else:
                instruction = f"Complete the task: {task_id}"

            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                ],
                "ground_truth": task_id,
                "dataset": "env_last",
                "env_config": {"task_id": task_id},
            }
            f.write(json.dumps(record) + "\n")

    logger.info(f"Wrote {len(task_ids)} records to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create agent_task datasets from various formats.")
    parser.add_argument("--input_dataset", type=str, help="HuggingFace dataset path (for tar/swe_gym formats)")
    parser.add_argument("--input_dir", type=str, help="Local directory path (for directory format)")
    parser.add_argument(
        "--input_format",
        type=str,
        required=True,
        choices=["tar", "directory", "swe_gym"],
        help="Input format",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for task data")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--build_images", action="store_true", help="Build Docker images from Dockerfiles")
    parser.add_argument("--registry", type=str, help="Registry prefix for pushing images (e.g., ghcr.io/myorg)")
    parser.add_argument("--default_image", type=str, help="Default Docker image for tasks without Dockerfiles")
    parser.add_argument("--output_file", type=str, help="Output JSONL file path (default: {output_dir}/dataset.jsonl)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract tasks based on format
    if args.input_format == "tar":
        if not args.input_dataset:
            parser.error("--input_dataset is required for tar format")
        task_ids = extract_tar_dataset(args.input_dataset, args.output_dir, args.split)
    elif args.input_format == "directory":
        if not args.input_dir:
            parser.error("--input_dir is required for directory format")
        task_ids = extract_directory_dataset(args.input_dir, args.output_dir)
    elif args.input_format == "swe_gym":
        if not args.input_dataset:
            parser.error("--input_dataset is required for swe_gym format")
        task_ids = extract_swe_gym_dataset(args.input_dataset, args.output_dir, args.split)
    else:
        parser.error(f"Unknown format: {args.input_format}")

    if not task_ids:
        logger.warning("No tasks found!")
        return

    # Build images if requested
    if args.build_images:
        build_and_push_images(args.output_dir, task_ids, args.registry)

    # Write default image for tasks without image.txt
    if args.default_image:
        write_default_images(args.output_dir, task_ids, args.default_image)

    # Generate JSONL dataset
    output_file = args.output_file or os.path.join(args.output_dir, "dataset.jsonl")
    generate_jsonl(args.output_dir, task_ids, output_file)

    logger.info(f"Done! {len(task_ids)} tasks processed. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
