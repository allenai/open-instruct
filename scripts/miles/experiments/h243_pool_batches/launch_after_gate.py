"""Wait for H243 mechanics, audit its receipts, then launch/register the pair.

Run on the authenticated submission host. No GPU is held while waiting. The
script is restartable: a durable receipt prevents duplicate arm submissions.
It leaves both draft PRs open and changes no registered metric or decision rule.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.execution import launch

ROOT = Path(__file__).resolve().parents[4]
CONFIGS = ROOT / "configs/miles/experiments/h243-pool-batches-20261116"


def command(*args, cwd=ROOT, env=None):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True)


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def audit(results):
    runs = list(results.glob("*/run"))
    if len(runs) != 1:
        raise RuntimeError("Expected one mechanics result tree")
    root = runs[0]
    contracts = sorted((root / "checkpoints").glob("training_contract_rank*.jsonl"))
    if len(contracts) != 2:
        raise RuntimeError("Expected both trainer-rank contracts")
    for path in contracts:
        optimizers = [json.loads(line) for line in path.read_text().splitlines() if line]
        optimizers = [row for row in optimizers if row.get("event") == "optimizer"]
        if [row["step"] for row in optimizers] != [1, 2, 3, 4]:
            raise RuntimeError(f"Incomplete or repeated successful updates in {path.name}")
        if any(row["reduction"] != "token" or row.get("optimizer_skipped") is not False for row in optimizers):
            raise RuntimeError("Mechanics did not use successful token-mean updates")
    manifests = sorted((root / "checkpoints/core").glob("rollout_*/complete.json"))
    steps = {json.loads(path.read_text())["clock"]["completed_steps"] for path in manifests}
    if not {2, 4} <= steps:
        raise RuntimeError(f"Missing retained mechanics checkpoints: {steps}")
    return {"successful_updates_per_rank": 4, "ranks": 2, "checkpoint_steps": sorted(steps)}


def record(ledger, arm, receipt, options, gate_url):
    if command("git", "status", "--porcelain", cwd=ledger).strip():
        raise RuntimeError("Ledger checkout must be clean before automatic registration")
    path = next((ledger / "hypotheses").glob("H243-*.md"))
    _, front, body = path.read_text().split("---", 2)
    document = yaml.safe_load(front)
    run = next((run for run in document["runs"] if run["arm"] == arm), None)
    if run is not None:
        if run["beaker"] != receipt["experiment_id"]:
            raise RuntimeError("Ledger already names a different experiment for this arm")
        return
    config_path = CONFIGS / f"{arm}.json"
    config = json.loads(config_path.read_text())
    lock = json.loads((ROOT / "runtime/miles/runtime.lock.json").read_text())
    commits = [
        f"allenai/open-instruct@{revision}"
        for revision in dict.fromkeys([options.image_revision, options.launch_revision])
    ]
    commits.extend(
        f"{source['repository'].removeprefix('https://github.com/').removesuffix('.git')}@{source['revision']}"
        for source in lock["sources"].values()
    )
    document["runs"].append(
        {
            "arm": arm,
            "beaker": receipt["experiment_id"],
            "seed": 17,
            "ckpt": config["output"]["root"] + "/checkpoints",
            "retention": "keep",
            "commit": commits,
            "launch": f"https://github.com/allenai/open-instruct/blob/{options.launch_revision}/{config_path.relative_to(ROOT)}",
            "wandb": None,
            "evals": [],
        }
    )
    document["status"] = "running"
    body = body.replace(
        "Status: registered before scientific GPU launch. No H243 training/evaluation result exists.",
        "Status: scientific training submitted after the mechanics gate passed. No H243 evaluation result exists.",
    )
    marker = "\nGPU gate: pending replacement"
    if marker in body:
        body = body[: body.index(marker)]
    # Gate details remain in the linked issue; the ledger carries one receipt line.
    if "GPU mechanics gate passed:" not in body:
        body += f"\nGPU mechanics gate passed: four token-mean optimizer updates on both ranks, with checkpoints at 2/4. [Receipt]({gate_url}).\n"
    path.write_text("---\n" + yaml.safe_dump(document, sort_keys=False, width=110) + "---" + body)
    command("python", "-m", "ledger.cli", "validate", cwd=ledger)
    command("git", "add", str(path.relative_to(ledger)), cwd=ledger)
    command("git", "commit", "-m", f"Record H243 {arm} scientific launch", cwd=ledger)
    command("git", "push", cwd=ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("smoke", "image", "image-revision", "launch-revision", "ledger", "state"):
        parser.add_argument("--" + name, required=True)
    options = parser.parse_args()
    ledger, state = Path(options.ledger), Path(options.state)
    state.mkdir(parents=True, exist_ok=True)
    if command("git", "rev-parse", "HEAD").strip() != options.launch_revision:
        raise RuntimeError("Submission checkout revision changed")
    if command("git", "status", "--porcelain").strip():
        raise RuntimeError("Submission checkout must remain clean")
    canceled_attempt, canceled_at = None, None
    while True:
        tasks = json.loads(command("beaker", "experiment", "tasks", options.smoke, "--format", "json"))
        if len(tasks) != 1:
            raise RuntimeError("Expected one mechanics task")
        by_rank = {}
        for job in sorted(tasks[0]["jobs"], key=lambda job: job["status"]["created"]):
            by_rank[job.get("execution", {}).get("replicaRank", 0)] = job
        if len(by_rank) != 1:
            raise RuntimeError("Expected one mechanics replica")
        current = next(iter(by_rank.values()))
        status = current["status"]
        write(state / "gate-status.json", {"job": current["id"], "status": status})
        if status.get("finalized"):
            # A healthcheck/preemption cancellation can acquire a newer attempt.
            # Give the scheduler three polling intervals before calling it final.
            if status.get("canceledCode") in (1, 2, 6, 9, 10, 11):
                if canceled_attempt != current["id"]:
                    canceled_attempt, canceled_at = current["id"], time.monotonic()
                if time.monotonic() - canceled_at < 135:
                    time.sleep(45)
                    continue
            if status.get("exitCode") != 0:
                raise RuntimeError(f"Mechanics failed or was canceled: {status}")
            break
        print(f"Waiting for mechanics: {current['id']} {status}", flush=True)
        time.sleep(45)
    results = state / "gate-results"
    command("beaker", "experiment", "results", options.smoke, "--output", str(results))
    gate = audit(results)
    write(state / "gate-audit.json", gate)
    comment = (
        "Drafted by Codex.\n\nGPU mechanics passed: "
        + json.dumps(gate)
        + f". [Beaker](https://beaker.org/ex/{options.smoke}). Submitting the registered scientific pair."
    )
    response = command(
        "gh", "api", "repos/allenai/olmo-post-training-ledger/issues/1394/comments", "-f", "body=" + comment
    )
    gate_url = json.loads(response)["html_url"]
    for arm in ("mixed-token", "pool-token"):
        path = state / f"{arm}-receipt.json"
        if not path.exists():
            if command("git", "rev-parse", "HEAD").strip() != options.launch_revision:
                raise RuntimeError("Submission revision changed while waiting")
            # The canonical submitter writes its receipt before returning. Recover
            # it if this controller stopped after submission but before its own write.
            spec = RunSpec.load(CONFIGS / f"{arm}.json")
            cached_path = launch.receipt_path(spec)
            if cached_path.exists():
                receipt = json.loads(cached_path.read_text())
                if (
                    receipt.get("revision") != options.launch_revision
                    or receipt.get("image") != options.image
                    or receipt.get("spec") != spec.to_dict()
                ):
                    raise RuntimeError("Existing canonical arm receipt has different provenance")
                write(path, receipt)
            else:
                env = os.environ | {"MILES_EXISTING_IMAGE": options.image}
                output = command(
                    "bash",
                    "./scripts/train/build_image_and_launch.sh",
                    "--miles",
                    str(CONFIGS / f"{arm}.json"),
                    env=env,
                )
                # submit prints the single JSON receipt; diagnostics go to stderr.
                receipt = json.loads(output)
                write(path, receipt)
        receipt = json.loads(path.read_text())
        record(ledger, arm, receipt, options, gate_url)
        print(f"Submitted {arm}: https://beaker.org/ex/{receipt['experiment_id']}", flush=True)
    links = "\n".join(
        f"{index}. {arm}: [Beaker](https://beaker.org/ex/{json.loads((state / f'{arm}-receipt.json').read_text())['experiment_id']})"
        for index, arm in enumerate(("mixed-token", "pool-token"), 1)
    )
    update = f"Drafted by Codex.\n\nH243 scientific pair submitted after the four-update mechanics gate passed.\n\nRuns:\n\n{links}"
    command("gh", "api", "repos/allenai/olmo-post-training-ledger/issues/1394/comments", "-f", "body=" + update)
    for repository, number in (("allenai/open-instruct", 2273), ("allenai/olmo-post-training-ledger", 1395)):
        body = json.loads(command("gh", "api", f"repos/{repository}/pulls/{number}"))["body"]
        status = f"\n\n<!-- H243 current launch status -->\nGPU mechanics passed; the registered scientific pair is submitted.\n\nRuns:\n\n{links}\n<!-- /H243 current launch status -->\n"
        command("gh", "api", "-X", "PATCH", f"repos/{repository}/pulls/{number}", "-f", "body=" + body + status)
    write(
        state / "complete.json",
        {
            "arms": [
                json.loads((state / f"{arm}-receipt.json").read_text())["experiment_id"]
                for arm in ("mixed-token", "pool-token")
            ]
        },
    )


if __name__ == "__main__":
    main()
