#!/usr/bin/env python3
"""Convert osieosie/tmax-tasks-* dataset to SWERL sandbox format.

Creates:
1. HuggingFace dataset with messages/ground_truth/dataset/env_config/source columns
2. Task data tarball with tests/test.sh, setup.sh, and instruction.md per task
"""

import argparse
import io
import json
import os
import tarfile

from datasets import Dataset, load_dataset

SYSTEM_PROMPT = (
    "You are a helpful coding assistant. You have access to a bash terminal. "
    "Use it to explore the codebase, understand the problem, implement a solution, "
    "and verify it works. When you are confident your solution is correct, submit "
    "by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
)


def parse_container_def(container_def: str) -> tuple[str, str]:
    """Extract base image and %post commands from a Singularity/Apptainer def."""
    image = "python:3.12-slim"
    post_commands = ""

    for line in container_def.splitlines():
        stripped = line.strip()
        if stripped.startswith("From:"):
            image = stripped.split(":", 1)[1].strip()

    in_post = False
    post_lines = []
    for line in container_def.splitlines():
        if line.strip() == "%post":
            in_post = True
            continue
        if line.strip().startswith("%") and in_post:
            break
        if in_post:
            post_lines.append(line)

    post_commands = "\n".join(post_lines).strip()
    return image, post_commands


def make_test_sh(test_final_state: str) -> str:
    """Create test.sh that writes the test file and runs pytest."""
    escaped = test_final_state.replace("'", "'\\''")
    return f"""#!/bin/bash
set -e
mkdir -p /logs/verifier

cat << 'TEST_EOF' > /tmp/test_final_state.py
{test_final_state}
TEST_EOF

if python3 -m pytest /tmp/test_final_state.py -x --tb=short 2>&1; then
    echo "1" > /logs/verifier/reward.txt
else
    echo "0" > /logs/verifier/reward.txt
fi
"""


def make_setup_sh(post_commands: str) -> str:
    """Create setup.sh from container %post commands."""
    return f"""#!/bin/bash
set -e
{post_commands}
"""


def convert(input_dataset: str, output_dataset: str, output_tarball: str, split: str = "train"):
    ds = load_dataset(input_dataset, split=split)
    print(f"Loaded {len(ds)} rows from {input_dataset}")

    records = []
    tar_stream = io.BytesIO()

    with tarfile.open(fileobj=tar_stream, mode="w:gz") as tar:
        for row in ds:
            task_id = row["task_id"]
            description = row["description"]
            test_final = row["test_final_state"]
            container_def = row["container_def"]

            image, post_commands = parse_container_def(container_def)

            # Create HF dataset record
            records.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": description},
                ],
                "ground_truth": task_id,
                "dataset": "passthrough",
                "env_config": {"env_name": "swerl_sandbox", "task_id": task_id, "image": image},
                "source": "tmax_skill_taxonomy",
            })

            # Add instruction.md
            _add_file(tar, f"{task_id}/instruction.md", description)

            # Add tests/test.sh
            test_sh = make_test_sh(test_final)
            _add_file(tar, f"{task_id}/tests/test.sh", test_sh)

            # Add setup.sh (container %post commands for package install + seed files)
            if post_commands:
                setup_sh = make_setup_sh(post_commands)
                _add_file(tar, f"{task_id}/setup.sh", setup_sh)

    tar_stream.seek(0)
    os.makedirs(os.path.dirname(output_tarball) or ".", exist_ok=True)
    with open(output_tarball, "wb") as f:
        f.write(tar_stream.read())
    print(f"Wrote task data to {output_tarball}")

    # Upload HF dataset
    hf_ds = Dataset.from_list(records)
    hf_ds.push_to_hub(output_dataset, split="train")
    print(f"Pushed {len(records)} records to {output_dataset}")


def _add_file(tar: tarfile.TarFile, path: str, content: str) -> None:
    data = content.encode("utf-8")
    info = tarfile.TarInfo(name=path)
    info.size = len(data)
    info.mode = 0o755
    tar.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description="Convert tmax-tasks dataset to SWERL format")
    parser.add_argument("--input", default="osieosie/tmax-tasks-skill-taxonomy-20260324-1k-verified")
    parser.add_argument("--output-dataset", default="hamishivi/swerl-tmax-skill-taxonomy-1k")
    parser.add_argument("--output-tarball", default="data/tmax-task-data.tar.gz")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    convert(args.input, args.output_dataset, args.output_tarball, args.split)


if __name__ == "__main__":
    main()
