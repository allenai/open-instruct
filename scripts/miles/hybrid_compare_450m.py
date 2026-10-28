"""One-time, fail-closed launcher for the approved 450m infrastructure comparison.

Run on Hammond with Beaker/GitHub credentials in the environment. This waits for
an already submitted smoke, validates its durable evidence, and submits each
500-update job once. An interrupted submission requires manual reconciliation.
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


def command(argv, *, cwd=None, env=None):
    return subprocess.check_output(argv, cwd=cwd, env=env, text=True, stderr=subprocess.STDOUT)


def validate_smoke(root):
    state = json.loads((root / "workflow.json").read_text())
    if state.get("status") != "complete" or state.get("result", {}).get("completed_rollout_ids") != [0, 1]:
        raise RuntimeError("Smoke did not complete exactly two learner updates")
    stages = [json.loads(line) for line in (root / "checkpoints/driver_timing.jsonl").read_text().splitlines()]
    for name, minimum in (("training", 2), ("publication", 2), ("evaluation", 3), ("checkpoint", 2)):
        if sum(row["stage"] == name and row.get("passed", False) for row in stages) < minimum:
            raise RuntimeError(f"Smoke lacks successful {name} evidence")
    for rollout in (0, 1):
        manifest = root / "checkpoints/core" / f"rollout_{rollout:07d}/complete.json"
        if not json.loads(manifest.read_text()).get("cursor_sha256"):
            raise RuntimeError("Smoke checkpoint lacks committed rollout cursor")
    if not (root / "export-hf/.complete").is_file():
        raise RuntimeError("Smoke final HF export is incomplete")


def save(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def submit_once(label, argv, repo, state_path, state, env):
    # Persist intent first: never resubmit after an unknown submission outcome.
    state.update(phase=f"submitting_{label}")
    save(state_path, state)
    log = state_path.parent / f"{label}-submit.log"
    with log.open("w") as stream:
        subprocess.run(argv, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    matches = re.findall(r"https://beaker.org/ex/([0-9A-Z]{26})", log.read_text())
    if not matches:
        raise RuntimeError(f"No experiment receipt in {log}; reconcile before retrying")
    experiment = matches[-1]
    state[label] = experiment
    save(state_path, state)
    spec = command(["beaker", "experiment", "spec", experiment])
    (state_path.parent / f"{label}-spec.yaml").write_text(spec)
    for expected in ("autoResume: true", "minRuntime: 0s", "gpuCount: 2", "ai2/rhea", "allowedTaskRetries: 3"):
        if expected not in spec:
            command(["beaker", "experiment", "stop", experiment])
            raise RuntimeError(f"Stopped {experiment}: submitted spec lacks {expected}")
    return experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-experiment", required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--miles-image", required=True)
    parser.add_argument("--miles-repo", type=Path, required=True)
    parser.add_argument("--core-repo", type=Path, required=True)
    parser.add_argument("--core-revision", required=True)
    parser.add_argument("--core-image", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.state_dir / "controller.json"
    # Exclusive creation makes repeated invocations fail closed.
    with state_path.open("x") as stream:
        json.dump({"phase": "starting"}, stream)
    state = {"phase": "waiting_for_smoke", "smoke": args.smoke_experiment, "sizes": ["450m"]}
    save(state_path, state)
    try:
        while True:
            tasks = json.loads(command(["beaker", "experiment", "tasks", args.smoke_experiment, "--format", "json"]))
            jobs = [job for task in tasks for job in task.get("jobs", [])]
            if jobs:
                status = jobs[-1]["status"]
                if status.get("canceled") or status.get("exited"):
                    if status.get("exitCode") != 0 or status.get("canceled"):
                        raise RuntimeError(f"Smoke failed: {status}")
                    break
                logs = command(["beaker", "experiment", "logs", args.smoke_experiment])
                if any(text in logs for text in ("Scheduler hit an exception:", "ray.exceptions.RayTaskError")):
                    command(["beaker", "experiment", "stop", args.smoke_experiment])
                    raise RuntimeError("Stopped smoke after fatal worker error; inspect Beaker logs")
            time.sleep(20)
        validate_smoke(args.smoke_root)
        for repo in (args.miles_repo, args.core_repo):
            if command(["git", "status", "--porcelain"], cwd=repo).strip():
                raise RuntimeError(f"Checkout is dirty: {repo}")
        if not command(["git", "rev-parse", "HEAD"], cwd=args.core_repo).startswith(args.core_revision):
            raise RuntimeError("Core comparison checkout changed; image must be reconciled")
        state["phase"] = "smoke_passed"
        save(state_path, state)
        env = dict(os.environ, MILES_EXISTING_IMAGE=args.miles_image, NO_COLOR="1", COLUMNS="240")
        submit_once(
            "miles",
            [
                "python",
                "-m",
                "open_instruct.miles",
                "run",
                "configs/miles/benchmarks/hybrid_450m_gsm8k_compare_500.toml",
            ],
            args.miles_repo,
            state_path,
            state,
            env,
        )
        alias = "michaeln/open-instruct-integration-test-hybrid-small-suite-cuda12"
        command(["beaker", "image", "rename", alias, ""])
        command(["beaker", "image", "rename", args.core_image, alias.split("/")[1]])
        env.update(
            MODEL_SIZE="450m",
            RUN_NAME="hybrid-infra-500-core-vllm-450m-20261027",
            SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPEN_INSTRUCT="0.0.1",
        )
        lockfile = args.core_repo / "uv.lock"
        before_sync = lockfile.read_bytes()
        try:
            submit_once(
                "core_vllm",
                [
                    "./scripts/train/build_image_and_launch.sh",
                    "scripts/train/olmo-hybrid-small/gsm8k_infra_compare_500.sh",
                ],
                args.core_repo,
                state_path,
                state,
                env,
            )
        finally:
            if lockfile.read_bytes() != before_sync:
                (args.state_dir / "core-after-sync-uv.lock").write_bytes(lockfile.read_bytes())
                lockfile.write_bytes(before_sync)
        state["phase"] = "both_submitted"
        save(state_path, state)
    except Exception as error:
        state.update(phase="failed", error=str(error))
        save(state_path, state)
        raise


if __name__ == "__main__":
    main()
