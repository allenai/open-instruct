"""Launch selected basket cases using committed overlays on an immutable image."""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from scripts.miles import throughput_basket as basket

from open_instruct.miles import launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("cases", nargs="+", choices=list(basket.CASES))
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--exclude-hostname", action="append", default=[], help="Exclude an unhealthy host from this basket only")
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=basket.ROOT):
        raise RuntimeError("Commit changes before launch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=basket.ROOT, text=True).strip()
    with tempfile.TemporaryDirectory(prefix="throughput-src-") as directory:
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
                    "tests/miles",
                    "configs/miles",
                    "runtime/miles/runtime.lock.json",
                ],
                cwd=basket.ROOT,
            )
        )
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        source = "SOURCE_DATASET"
        provenance = dict(commit=commit, base_image=args.image, archive_sha256=digest, excluded_hostnames=args.exclude_hostname)
        (directory / "provenance.json").write_text(json.dumps(provenance, indent=2))
        if not args.render_only:
            name = "throughput-source-" + uuid.uuid4().hex[:12]
            subprocess.run(
                [
                    "beaker",
                    "dataset",
                    "create",
                    str(directory),
                    "--workspace",
                    "ai2/open-instruct-dev",
                    "--budget",
                    "ai2/oe-other",
                    "--name",
                    name,
                ],
                check=True,
                stdout=sys.stderr,
            )
            author = json.loads(
                subprocess.check_output(["beaker", "account", "whoami", "--format", "json"], text=True)
            )[0]["name"]
            source = json.loads(
                subprocess.check_output(
                    ["beaker", "dataset", "get", f"{author}/{name}", "--format", "json"], text=True
                )
            )[0]["id"]
        for case in args.cases:
            identity = uuid.uuid4().hex[:12]
            root = (
                Path("/weka/oe-training-default/robertb/open-instruct/throughput-profiles")
                / (case + "-" + identity)
                / "run"
            )
            run = basket.specification(case, root)
            hostnames = [f"host-{i}" for i in range(32)] if args.render_only else None
            if args.exclude_hostname:
                hostnames = hostnames if hostnames is not None else launch.cluster_hostnames(run)
                hostnames = [name for name in hostnames if name not in args.exclude_hostname]
                if not hostnames:
                    raise ValueError("No eligible hosts remain after exclusions")
            spec = launch.specification(args.image, run, hostnames=hostnames)
            if args.exclude_hostname and len(spec["tasks"]) == 1:
                spec["tasks"][0]["constraints"] = {"hostname": hostnames}
            overlay = f"""mkdir -p /output
cp /qualification-source/provenance.json /output/
echo '{digest}  /qualification-source/source.tar' | sha256sum -c -
tar -xf /qualification-source/source.tar -C /opt/core-rl
git -C /opt/core-rl/sources/miles apply --check /opt/core-rl/scripts/miles/diagnostics/policy-refresh-runtime.patch
git -C /opt/core-rl/sources/miles apply /opt/core-rl/scripts/miles/diagnostics/policy-refresh-runtime.patch
"""
            if case in ("dev", "tiny"):
                overlay += (
                    "python tests/miles/local_moe.py bootstrap " + shlex.quote(str(root.parent / "fixture")) + "\n"
                )
            for index, task in enumerate(spec["tasks"]):
                task["datasets"].append(dict(mountPath="/qualification-source", source=dict(beaker=source)))
                task["arguments"][0] = task["arguments"][0].replace(
                    "cd /opt/core-rl\n", "cd /opt/core-rl\n" + overlay, 1
                )
                if index == 0:
                    task["arguments"][0] += (
                        "\npython -m scripts.miles.throughput_basket "
                        + shlex.quote(str(root))
                        + (" --warmup 1" if case in ("dev", "tiny") else "")
                        + "\n"
                    )
                trainer_ranks = run.plan()["allocation"]["nodes"][index]["trainer_gpus"]
                task["resources"]["memory"] = (
                    "64 GiB" if case in ("dev", "tiny") else f"{max(256, 64 + 80 * trainer_ranks)} GiB"
                )
                task["resources"]["cpuCount"] = 24 if case in ("dev", "tiny") else 32
                task["resources"]["sharedMemory"] = "32 GiB" if case in ("dev", "tiny") else "128 GiB"
            if args.render_only:
                print(json.dumps(dict(case=case, output=str(root), specification=spec), indent=2))
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
                        "--format",
                        "json",
                    ],
                    text=True,
                )
            )
            print(
                json.dumps(dict(case=case, output=str(root), experiment=result[0]["id"], source=source, **provenance)),
                flush=True,
            )


if __name__ == "__main__":
    main()
