"""Launch committed refresh sources over the immutable qualification image."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from scripts.miles.launch_gsm8k_parity import ROOT as CAMPAIGN_ROOT

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--mode", choices=("refresh", "barrier"), default="refresh")
    parser.add_argument("--render-only", action="store_true")
    opt = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise RuntimeError("Commit source before creating the qualification artifact")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with tempfile.TemporaryDirectory(prefix="refresh-trial-") as temporary:
        root = Path(temporary)
        archive = root / "source.tar"
        with archive.open("wb") as stream:
            subprocess.run(
                [
                    "git",
                    "archive",
                    "HEAD",
                    "open_instruct/miles",
                    "scripts/miles",
                    "tests/miles",
                    "runtime/miles/runtime.lock.json",
                ],
                cwd=ROOT,
                stdout=stream,
                check=True,
            )
        provenance = dict(
            commit=commit,
            image=opt.image,
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            runtime_parent="bc582bc5c680b2138d50cda141322155207a34fa",
            scope="Committed Open-Instruct overlay and checked MILES delta; no floating source fetch",
        )
        (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        dataset = "SOURCE_DATASET"
        if not opt.render_only:
            name = f"policy-refresh-src-{commit[:12]}-{uuid.uuid4().hex[:8]}"
            subprocess.run(
                [
                    "beaker",
                    "dataset",
                    "create",
                    str(root),
                    "--workspace",
                    "ai2/open-instruct-dev",
                    "--budget",
                    "ai2/oe-other",
                    "--name",
                    name,
                    "--desc",
                    f"Policy refresh qualification source {commit}",
                ],
                check=True,
                stdout=sys.stderr,
            )
            author = json.loads(
                subprocess.check_output(["beaker", "account", "whoami", "--format", "json"], text=True)
            )[0]["name"]
            uploaded = subprocess.check_output(
                ["beaker", "dataset", "get", f"{author}/{name}", "--format", "json"], text=True
            )
            dataset = json.loads(uploaded)[0]["id"]
        mode = opt.mode
        command = f"""set -euo pipefail
cd /opt/core-rl
mkdir -p /output
cp /qualification-source/provenance.json /output/
echo '{provenance["archive_sha256"]}  /qualification-source/source.tar' | sha256sum -c -
tar -xf /qualification-source/source.tar
cd /opt/core-rl/sources/miles
git apply --check /opt/core-rl/scripts/miles/diagnostics/policy-refresh-runtime.patch
git apply /opt/core-rl/scripts/miles/diagnostics/policy-refresh-runtime.patch
cd /opt/core-rl
export RUN_ROOT=/weka/oe-training-default/robertb/open-instruct/policy-refresh/$BEAKER_EXPERIMENT_ID/{mode}
export SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
python -m pytest tests/miles/test_policy_refresh_runtime.py tests/miles/test_async_publication_boundary.py -q
python -m scripts.miles.policy_refresh_trial {CAMPAIGN_ROOT} "$RUN_ROOT" --mode {mode} --updates 4 2>&1 | tee /output/initial.log
python -m scripts.miles.policy_refresh_trial {CAMPAIGN_ROOT} "$RUN_ROOT" --mode {mode} --updates 5 --resume 2>&1 | tee /output/resume.log
cp "$RUN_ROOT/"*-result.json /output/
"""
        spec = dict(
            version="v2",
            budget="ai2/oe-other",
            description=f"Actual Core RL {mode}: EP2 plus two TP1 engines, replay, save/resume and eval",
            tasks=[
                dict(
                    name=f"policy-{mode}-rl",
                    image={"beaker": opt.image},
                    command=["bash", "-lc"],
                    arguments=[command],
                    resources=dict(gpuCount=4, cpuCount=24, memory="256 GiB", sharedMemory="128 GiB"),
                    context=dict(priority="urgent", minRuntime="1h", autoResume=False),
                    constraints=dict(cluster=["ai2/holmes"]),
                    timeout="90m",
                    result=dict(path="/output"),
                    datasets=[
                        dict(mountPath="/qualification-source", source=dict(beaker=dataset)),
                        dict(mountPath="/weka/oe-training-default", source=dict(weka="oe-training-default")),
                    ],
                    envVars=[dict(name="NCCL_CUMEM_ENABLE", value="1"), dict(name="OMP_NUM_THREADS", value="2")],
                )
            ],
        )
        if opt.render_only:
            print(json.dumps(spec, indent=2))
            return
        path = root / "experiment.json"
        path.write_text(json.dumps(spec))
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
