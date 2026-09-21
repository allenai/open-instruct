"""Standalone olmo-eval worker and manual W&B publisher.

The submitter transports this file into a pinned evaluator image. That image must
contain /opt/olmo-eval at the requested revision, olmo-eval, W&B, and the qualified
SGLang/olmo-sglang runtime. No training imports or credentials are transported.
"""

import argparse
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request


class PublicationBlocked(RuntimeError):
    """The candidate writer failed its live run-lifecycle qualification."""


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def result_directory(receipt):
    names = "\n".join(task["task"] for task in receipt["tasks"])
    # The submitter gives each task group a stable, collision-resistant ID.
    key = receipt["group_id"]
    assert names
    return Path(receipt["evaluation"]["root"]) / "evaluation" / "results" / f"update-{receipt['update']:08d}-{key}"


def command(receipt, output):
    checkpoint = receipt["checkpoint"]
    args = ["olmo-eval", "run", "--model", checkpoint, "--harness", "default"]
    for key, value in {
        "provider.kind": "vllm_server",
        "provider.base_url": "http://127.0.0.1:30000/v1",
        "provider.tokenizer": checkpoint,
        "provider.trust_remote_code": True,
        "provider.max_concurrency": 8,
    }.items():
        args.extend(["-o", f"{key}={value if isinstance(value, str) else json.dumps(value)}"])
    for task in receipt["tasks"]:
        args.extend(["--task", task["task"]])
        for key, value in task["generation"].items():
            args.extend(["-o", f"{key}={value if isinstance(value, str) else json.dumps(value)}"])
        for key, value in task["scoring"].items():
            args.extend(["-o", f"{key}={value if isinstance(value, str) else json.dumps(value)}"])
    return [*args, "--output-dir", str(output), "--save-predictions", "--save-requests"]


def scores(output):
    """Read the pinned olmo-eval metrics schema, without treating failures as zero."""
    files = list(Path(output).rglob("metrics.json"))
    if not files:
        raise ValueError("No olmo-eval metrics.json found")
    values = {}
    for path in files:
        payload = json.loads(path.read_text())
        for task in payload["tasks"]:
            if task.get("error_summary") or task.get("instances_saved") == 0:
                continue
            for metric, scorers in task["metrics"].items():
                for scorer, score in scorers.items():
                    if type(score) in (int, float) and math.isfinite(score):
                        key = f"eval/{task['task']}/{metric}/{scorer}"
                        if key in values:
                            raise ValueError(f"Duplicate metric {key}")
                        values[key] = score
    if not values:
        raise ValueError("No successful numeric evaluation scores")
    return values


def require_safe_shared_writer():
    # Live qualification with W&B 0.30.0 reopened finished runs and replaced
    # their metric definitions despite the documented secondary-writer flags.
    # Keep the candidate publisher disabled until that lifecycle contract is
    # supported. Restoring state after upload would race the training writer.
    raise PublicationBlocked(
        "W&B shared-writer lifecycle qualification failed: late uploads reopen finished runs "
        "and replace metric definitions. Results retained; publication is blocked. "
        "See docs/miles/measurements/background-evaluation-20260920.md"
    )


def publish(receipt, output, *, wandb_run=None):
    tracking = dict(receipt["training"]["wandb"])
    if wandb_run:
        tracking["entity"], tracking["project"], tracking["id"] = wandb_run.split("/")
    elif tracking["mode"] != "online":
        write_json(
            Path(output) / "publication.json",
            {"status": "deferred", "reason": "Training offline or disabled; publish manually after sync"},
        )
        return
    if not all(tracking.get(key) for key in ("id", "entity", "project")):
        raise ValueError("Publishing requires the exact training entity/project/run ID")
    require_safe_shared_writer()
    values = scores(output)
    wandb = importlib.import_module("wandb")
    # Check existence before attaching: a typo must not create a different run.
    wandb.Api(timeout=30).run(f"{tracking['entity']}/{tracking['project']}/{tracking['id']}")
    run = wandb.init(
        id=tracking["id"],
        entity=tracking["entity"],
        project=tracking["project"],
        settings=wandb.Settings(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
            x_label=f"evaluation-{receipt['update']}-{receipt['group_id']}",
            init_timeout=60,
        ),
    )
    try:
        run.define_metric("eval/checkpoint_update")
        # Define exact names so other MILES writers' eval/* wildcard cannot change the axis.
        for key in values:
            run.define_metric(key, step_metric="eval/checkpoint_update", step_sync=False)
        run.log({"eval/checkpoint_update": receipt["update"], **values})
    finally:
        # Flush this secondary writer only. x_update_finish_state=False preserves
        # the primary run's running/finished/failed state, including late uploads.
        run.finish()
    write_json(
        Path(output) / "publication.json", {"status": "published", "run": tracking, "update": receipt["update"]}
    )


def try_publish(receipt, output, **kwargs):
    try:
        publish(receipt, output, **kwargs)
    except Exception as error:
        diagnostic = {"status": "failed", "error": type(error).__name__}
        if isinstance(error, PublicationBlocked):
            diagnostic.update(status="blocked", reason=str(error))
        write_json(Path(output) / "publication.json", diagnostic)
        print(
            f"WARNING: W&B publishing failed ({diagnostic.get('reason', type(error).__name__)}); "
            f"evaluation results retained at {output}",
            file=sys.stderr,
        )
        return False
    return True


def run_evaluation(receipt):
    output = result_directory(receipt)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "provenance.json", receipt)
    server = None
    try:
        revision = subprocess.check_output(
            ["git", "-C", "/opt/olmo-eval", "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
        if revision != receipt["evaluation"]["revision"]:
            raise ValueError("Evaluator revision does not match the pinned receipt")
        checkpoint = receipt["checkpoint"]
        for name in ("config.json", "tokenizer_config.json"):
            if not (Path(checkpoint) / name).is_file():
                raise ValueError(f"Incomplete HF snapshot: missing {name}")
        env = {**os.environ, "SGLANG_EXTERNAL_MODEL_PACKAGE": "olmo_sglang.models"}
        server_args = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            checkpoint,
            "--trust-remote-code",
            "--tokenizer-path",
            checkpoint,
            "--host",
            "127.0.0.1",
            "--port",
            "30000",
            "--tp-size",
            str(receipt["evaluation"]["gpus"]),
            *receipt["evaluation"]["server_args"],
        ]
        with (output / "server.log").open("w") as log:
            server = subprocess.Popen(server_args, env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 600
        while True:
            if server.poll() is not None:
                raise RuntimeError("Evaluator SGLang server exited during startup; see server.log")
            try:
                with request.urlopen("http://127.0.0.1:30000/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (urlerror.URLError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Evaluator SGLang startup timeout")
            time.sleep(1)
        args = command(receipt, output)
        write_json(output / "command.json", args)
        with (output / "olmo-eval.log").open("w") as log:
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
        aggregate = scores(output)
        write_json(output / "scores.json", aggregate)
        write_json(output / "evaluation.json", {"status": "complete"})
        try_publish(receipt, output)
    except Exception as error:
        write_json(output / "evaluation.json", {"status": "failed", "error": type(error).__name__})
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=15)
        # Preserve logs/results even when evaluation or publication failed.
        shutil.copytree(output, "/output/evaluation", dirs_exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "publish"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--wandb-run", help="entity/project/run_id, after offline training has been synced")
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if args.command == "run":
        run_evaluation(receipt)
    elif not try_publish(receipt, args.results or result_directory(receipt), wandb_run=args.wandb_run):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
