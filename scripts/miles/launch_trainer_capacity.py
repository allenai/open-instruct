"""Launch matched EP2 trainer screens on retained batches, without serving waits."""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from scripts.miles import trainer_capacity_config
from scripts.miles.throughput_basket import ROOT


def specification(image, source, digest, variant, output, commit):
    target = shlex.quote(str(output))
    kernel_environment = shlex.join(
        [f"{key}={value}" for key, value in trainer_capacity_config.environment(variant).items()]
    )
    command = f"""set -euo pipefail
cd /opt/core-rl
echo '{digest}  /qualification-source/source.tar' | sha256sum -c -
tar -xf /qualification-source/source.tar -C /opt/core-rl
git -C /opt/core-rl/sources/miles apply --check /opt/core-rl/scripts/miles/diagnostics/policy-refresh-runtime.patch
git -C /opt/core-rl/sources/miles apply /opt/core-rl/scripts/miles/diagnostics/policy-refresh-runtime.patch
mkdir -p {target} /output
cp /qualification-source/provenance.json /output/
export PYTHONPATH=/opt/core-rl:/opt/core-rl/sources/olmo-core/src:/opt/core-rl/sources/miles:/opt/core-rl/sources/olmo-sglang/src
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_CUMEM_ENABLE=1
export {kernel_environment}
python -m scripts.miles.sample_gpu_usage {shlex.quote(str(output / "gpu_usage.jsonl"))} &
trap 'cp {target}/*.json* /output/ 2>/dev/null || true; cp {target}/checkpoints/*.jsonl /output/ 2>/dev/null || true' EXIT
torchrun --standalone --nproc-per-node=2 --no-python bash scripts/miles/trainer_capacity_rank.sh --variant {shlex.quote(variant)} --output {target}
"""
    return dict(
        version="v2",
        budget="ai2/oe-other",
        description=f"Matched trainer capacity {variant}; source {commit}; isolated cold per-rank caches",
        tasks=[
            dict(
                name=output.name,
                image=dict(beaker=image),
                command=["bash", "-c"],
                arguments=[command],
                datasets=[
                    dict(mountPath="/qualification-source", source=dict(beaker=source)),
                    dict(mountPath="/weka/oe-training-default", source=dict(weka="oe-training-default")),
                ],
                result=dict(path="/output"),
                resources=dict(gpuCount=2, cpuCount=32, memory="256 GiB", sharedMemory="64 GiB"),
                context=dict(priority="urgent", minRuntime="1h", autoResume=False),
                constraints=dict(cluster=["ai2/holmes"]),
                timeout="2h",
            )
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(trainer_capacity_config.VARIANTS),
        default=list(trainer_capacity_config.VARIANTS),
    )
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise RuntimeError("Commit changes before launch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    identity = "trainer-capacity-" + uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory(prefix="trainer-capacity-source-") as directory:
        directory = Path(directory)
        archive = directory / "source.tar"
        archive.write_bytes(
            subprocess.check_output(
                [
                    "git",
                    "archive",
                    "HEAD",
                    "open_instruct/miles",
                    "scripts/miles",
                    "configs/miles",
                    "runtime/miles",
                    "tests/miles",
                ],
                cwd=ROOT,
            )
        )
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (directory / "provenance.json").write_text(
            json.dumps(
                dict(
                    commit=commit,
                    image=args.image,
                    archive_sha256=digest,
                    retained_batches=trainer_capacity_config.SOURCE,
                )
            )
        )
        source = "SOURCE_DATASET"
        if not args.render_only:
            subprocess.run(
                [
                    "beaker",
                    "dataset",
                    "create",
                    str(directory),
                    "--name",
                    identity,
                    "--workspace",
                    "ai2/open-instruct-dev",
                    "--budget",
                    "ai2/oe-other",
                ],
                check=True,
                stdout=sys.stderr,
            )
            author = json.loads(subprocess.check_output(["beaker", "account", "whoami", "--format=json"]))[0]["name"]
            source = json.loads(
                subprocess.check_output(["beaker", "dataset", "get", f"{author}/{identity}", "--format=json"])
            )[0]["id"]
        for variant in args.variants:
            output = (
                Path("/weka/oe-training-default/robertb/open-instruct/throughput-profiles")
                / f"trainer-{variant}-{uuid.uuid4().hex[:12]}"
            )
            spec = specification(args.image, source, digest, variant, output, commit)
            if args.render_only:
                print(json.dumps(spec))
                continue
            path = directory / "experiment.json"
            path.write_text(json.dumps(spec))
            result = json.loads(
                subprocess.check_output(
                    [
                        "beaker",
                        "experiment",
                        "create",
                        str(path),
                        "--workspace",
                        "ai2/open-instruct-dev",
                        "--format=json",
                    ]
                )
            )
            print(
                json.dumps(
                    dict(
                        variant=variant,
                        experiment=result[0]["id"],
                        source=source,
                        output=str(output),
                        commit=commit,
                        base_image=args.image,
                        archive_sha256=digest,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
