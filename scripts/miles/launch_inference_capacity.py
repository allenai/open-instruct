"""Launch one frozen-policy inference capacity sweep without trainer backpressure."""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from scripts.miles.throughput_basket import CAMPAIGN, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise RuntimeError("Commit changes before launch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    identity = "inference-capacity-" + uuid.uuid4().hex[:12]
    output = Path("/weka/oe-training-default/robertb/open-instruct/throughput-profiles") / identity
    with tempfile.TemporaryDirectory(prefix="capacity-source-") as directory:
        directory = Path(directory)
        archive = directory / "source.tar"
        archive.write_bytes(
            subprocess.check_output(
                ["git", "archive", "HEAD", "scripts/miles/inference_capacity.py", "scripts/miles/sample_gpu_usage.py"],
                cwd=ROOT,
            )
        )
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
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
        command = f"""set -euo pipefail
cd /opt/core-rl
echo '{digest}  /qualification-source/source.tar' | sha256sum -c -
tar -xf /qualification-source/source.tar -C /opt/core-rl
mkdir -p {shlex.quote(str(output))} /output
export PYTHONPATH=/opt/core-rl:/opt/core-rl/sources/olmo-sglang/src
export SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR={shlex.quote(str(output / "tmp-7d/triton"))}
python -m scripts.miles.sample_gpu_usage {shlex.quote(str(output / "gpu_usage.jsonl"))} &
trap 'cp {shlex.quote(str(output))}/*.json* /output/ 2>/dev/null || true' EXIT
python -m scripts.miles.inference_capacity --model {shlex.quote(str(CAMPAIGN / "hf"))} --output {shlex.quote(str(output))}
"""
        spec = dict(
            version="v2",
            budget="ai2/oe-other",
            description=f"Frozen TP1 inference capacity sweep; source {commit}",
            tasks=[
                dict(
                    name=identity,
                    image=dict(beaker=args.image),
                    command=["bash", "-c"],
                    arguments=[command],
                    datasets=[
                        dict(mountPath="/qualification-source", source=dict(beaker=source)),
                        dict(mountPath="/weka/oe-training-default", source=dict(weka="oe-training-default")),
                    ],
                    result=dict(path="/output"),
                    resources=dict(gpuCount=1, cpuCount=24, memory="256 GiB", sharedMemory="64 GiB"),
                    context=dict(priority="urgent", minRuntime="1h", autoResume=False),
                    constraints=dict(cluster=["ai2/holmes"]),
                    timeout="2h",
                )
            ],
        )
        if args.render_only:
            print(json.dumps(spec, indent=2))
            return
        path = directory / "experiment.json"
        path.write_text(json.dumps(spec))
        result = json.loads(
            subprocess.check_output(
                ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format=json"]
            )
        )
        print(
            json.dumps(
                dict(
                    experiment=result[0]["id"],
                    source=source,
                    output=str(output),
                    commit=commit,
                    base_image=args.image,
                    archive_sha256=digest,
                )
            )
        )


if __name__ == "__main__":
    main()
