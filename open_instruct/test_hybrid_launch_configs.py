"""CPU checks for the approved GPU split and the local-build safety boundary."""

import json
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

import mason

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("learners,engines", [(1, 1), (2, 2)])
def test_comparison_launcher_emits_exact_gpu_split(tmp_path, learners, engines):
    binary = tmp_path / "uv"
    output = tmp_path / "arguments.json"
    binary.write_text(
        f"#!{sys.executable}\nimport json,os,sys\nopen(os.environ['CAPTURE_ARGS'],'w').write(json.dumps(sys.argv[1:]))\n"
    )
    binary.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        CAPTURE_ARGS=str(output),
        MODEL_SIZE="450m",
        NUM_LEARNERS=str(learners),
        NUM_INFERENCE_ENGINES=str(engines),
    )
    subprocess.run(
        ["bash", "scripts/train/olmo-hybrid-small/gsm8k_infra_compare_500.sh", "immutable-image"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    args = json.loads(output.read_text())
    for flag, expected in (
        ("--gpus", learners + engines),
        ("--fsdp_shard_degree", learners),
        ("--num_learners_per_node", learners),
        ("--vllm_num_engines", engines),
        ("--total_episodes", 8000),
        ("--max_result_age_steps", 1),
    ):
        assert args[args.index(flag) + 1] == str(expected)


def test_existing_image_only_cannot_fall_through_to_a_local_build(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    wrapper = repo / "build.sh"
    shutil.copyfile(ROOT / "scripts/train/build_image_and_launch.sh", wrapper)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "add", "build.sh"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    beaker = binaries / "beaker"
    beaker.write_text('#!/bin/sh\nif [ "$1" = account ]; then echo \'[{"name":"michaeln"}]\'; else exit 1; fi\n')
    beaker.chmod(0o755)
    marker = tmp_path / "docker-was-called"
    docker = binaries / "docker"
    docker.write_text(f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\n")
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", "build.sh", "--existing-image-only", "unused.sh"],
        cwd=repo,
        env=dict(os.environ, PATH=f"{binaries}:{os.environ['PATH']}"),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "build it on Beaker first" in result.stdout
    assert not marker.exists()


def test_smoke_can_be_preemptible_without_automatically_restarting():
    args = Namespace(
        num_nodes=1,
        cluster=["ai2/rhea"],
        hostname=None,
        task_name="smoke",
        image="image",
        beaker_datasets=[],
        mount_docker_socket=False,
        priority="normal",
        preemptible=True,
        no_auto_resume=True,
        pure_docker_mode=True,
        env=[],
        secret=[],
        gpus=4,
        shared_memory="1gb",
        timeout="1h",
        no_host_networking=True,
    )
    task = mason.make_task_spec(args, "true", 0, [], "michaeln", False)
    assert task.context.min_runtime == 0
    assert task.context.auto_resume is False
    assert task.resources.gpu_count == 4
