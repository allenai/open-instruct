"""Standalone olmo-eval worker and manual W&B publisher.

The submitter transports this file into a pinned evaluator image. That image must
contain /opt/olmo-eval at the requested revision, olmo-eval, W&B, and the qualified
SGLang/olmo-sglang runtime. No training imports or credentials are transported.
"""

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib import error as urlerror
from urllib import request


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # This file is transported alone; keep publication independent of package imports.
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    stream = temporary.open("x")
    try:
        with stream:
            json.dump(value, stream, indent=2, sort_keys=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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


def result_roots(output):
    """Read only committed task attempts, not abandoned or incomplete attempts."""
    output = Path(output)
    index = output / "task-results.json"
    if not index.exists():
        return [output]
    roots = [output / name for name in json.loads(index.read_text())]
    if not roots or any(not root.resolve().is_relative_to(output.resolve()) for root in roots):
        raise ValueError("Invalid evaluation task-results paths")
    return roots


def validate_predictions(output):
    """Reject provider failures that the harness can encode as empty output lists."""
    paths = [path for root in result_roots(output) for path in root.rglob("*-predictions.jsonl")]
    if not paths:
        raise ValueError("No evaluation predictions found")
    for path in paths:
        count = 0
        with path.open() as stream:
            for count, line in enumerate(stream, 1):
                row = json.loads(line)
                if not row.get("model_output"):
                    raise ValueError(f"Missing model outputs in {path.name} row {count}; inspect evaluator logs")
        if count == 0:
            raise ValueError(f"Empty evaluation predictions: {path.name}")


def scores(output):
    """Read the pinned olmo-eval metrics schema, without treating failures as zero."""
    files = [path for root in result_roots(output) for path in root.rglob("metrics.json")]
    if not files:
        raise ValueError("No olmo-eval metrics.json found")
    if any(path for root in result_roots(output) for path in root.rglob("*-predictions.jsonl")):
        validate_predictions(output)
    values = {}
    for path in files:
        payload = json.loads(path.read_text())
        for task in payload["tasks"]:
            if task.get("error_summary") or task.get("instances_failed") or task.get("instances_saved") == 0:
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


def define_metrics(run):
    """Every MILES writer declares the same axes; W&B replaces this metadata.

    Include the native training/rollout groups, and use one wildcard for all
    background tasks so concurrent task groups cannot erase each other's axes.
    """
    for step in ("train/step", "rollout/step", "eval/step", "eval/checkpoint_update"):
        run.define_metric(step)
    for prefix, step in {
        "train": "train/step",
        "rollout": "rollout/step",
        "multi_turn": "rollout/step",
        "passrate": "rollout/step",
        "perf": "rollout/step",
        "eval": "eval/checkpoint_update",
    }.items():
        run.define_metric(f"{prefix}/*", step_metric=step, step_sync=prefix != "eval")


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
        define_metrics(run)
        run.log({"eval/checkpoint_update": receipt["update"], **values})
    finally:
        # Flush this writer without declaring the training run finished. A late
        # attachment can change the dashboard status back to running (accepted).
        run.finish()
    write_json(
        Path(output) / "publication.json", {"status": "published", "run": tracking, "update": receipt["update"]}
    )


def try_publish(receipt, output, **kwargs):
    try:
        publish(receipt, output, **kwargs)
    except Exception as error:
        diagnostic = {"status": "failed", "error": type(error).__name__}
        write_json(Path(output) / "publication.json", diagnostic)
        print(
            f"WARNING: W&B publishing failed ({diagnostic.get('reason', type(error).__name__)}); "
            f"evaluation results retained at {output}",
            file=sys.stderr,
        )
        return False
    return True


def resume_identity(receipt):
    """Scheduling and publication metadata may change; evaluated inputs may not."""
    config = receipt["evaluation"]
    return {
        **{key: receipt[key] for key in ("checkpoint", "update", "group_id", "tasks")},
        "evaluator": {key: config.get(key) for key in ("revision", "image", "gpus", "server_args", "worker")},
    }


@contextmanager
def evaluation_output(receipt):
    output = result_directory(receipt)
    output.mkdir(parents=True, exist_ok=True)
    # A kernel-owned lock is released after SIGKILL/preemption, unlike a PID file.
    # Never unlink it: another process may already have the same inode open.
    with (output / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another evaluator owns {output}") from error
        provenance = output / "provenance.json"
        if provenance.exists():
            if resume_identity(json.loads(provenance.read_text())) != resume_identity(receipt):
                raise ValueError("Existing evaluation results belong to different inputs; use a fresh output path")
        elif any(path.name != ".runner.lock" for path in output.iterdir()):
            raise ValueError("Existing evaluation output has no provenance; refusing to overwrite it")
        else:
            write_json(provenance, receipt)
        yield output


def completed_scores(output, tasks):
    """A success marker alone cannot establish that every requested task finished."""
    validate_predictions(output)
    values = scores(output)
    missing = [task["task"] for task in tasks if not any(k.startswith(f"eval/{task['task']}/") for k in values)]
    if missing:
        raise ValueError(f"Evaluation has no successful scores for requested tasks: {missing}")
    return values


def saved_task(output, task):
    key = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:16]
    directory = output / "tasks" / key
    marker = directory / "complete.json"
    if not marker.exists():
        return directory, None
    saved = json.loads(marker.read_text())
    if saved["task"] != task:
        raise ValueError("Completed task provenance mismatch")
    attempt = directory / saved["attempt"]
    if not attempt.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Invalid completed task path")
    if completed_scores(attempt, [task]) != saved["scores"]:
        raise ValueError("Completed task scores changed")
    return directory, attempt


def run_tasks(receipt, output):
    completed = []
    for task in receipt["tasks"]:
        directory, attempt = saved_task(output, task)
        if attempt is None:
            # olmo-eval has no response-level resume contract here. Keep partial
            # attempts intact and rerun only this unfinished task in a fresh path.
            attempt = directory / "attempts" / uuid.uuid4().hex
            attempt.mkdir(parents=True)
            args = command({**receipt, "tasks": [task]}, attempt)
            write_json(attempt / "command.json", args)
            with (attempt / "olmo-eval.log").open("w") as log:
                subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
            values = completed_scores(attempt, [task])
            write_json(
                directory / "complete.json",
                {"task": task, "attempt": str(attempt.relative_to(directory)), "scores": values},
            )
        else:
            print(f"Reusing completed evaluation task: {task['task']}", flush=True)
        completed.append(str(attempt.relative_to(output)))
    write_json(output / "task-results.json", completed)


def run_evaluation(receipt):
    with evaluation_output(receipt) as output:
        run_evaluation_locked(receipt, output)


def run_evaluation_locked(receipt, output):
    server = None
    try:
        status = output / "evaluation.json"
        if status.exists() and json.loads(status.read_text()).get("status") == "complete":
            completed_scores(output, receipt["tasks"])
            print(f"Reusing completed evaluation: {output}", flush=True)
            try_publish(receipt, output)
            return
        # Discover completions before starting an expensive inference server.
        pending = [task for task in receipt["tasks"] if saved_task(output, task)[1] is None]
        if not pending:
            run_tasks(receipt, output)
            write_json(output / "scores.json", completed_scores(output, receipt["tasks"]))
            write_json(output / "evaluation.json", {"status": "complete"})
            try_publish(receipt, output)
            return
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
        with (output / "server.log").open("a") as log:
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
        run_tasks(receipt, output)
        aggregate = completed_scores(output, receipt["tasks"])
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
