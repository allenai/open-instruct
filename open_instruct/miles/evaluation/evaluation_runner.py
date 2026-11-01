"""Run olmo-eval tasks for a submitted policy snapshot and publish their metrics.
The submitter copies this standalone worker into the pinned evaluator image, where
it validates output provenance, reuses completed task results after a restart and
isolates partial attempts. It invokes the installed evaluator and qualified serving
runtime without importing the training application, and can retry W&B publication
from saved results.
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
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib import error as urlerror
from urllib import request


def write_json(path, value):
    """Persist resumable receipts with the shared artifact helper's durability guarantees."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # This file is transported alone; keep publication independent of package imports.
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    stream = temporary.open("x")
    try:
        with stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
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
    first = receipt["tasks"][0]
    if first.get("kind", "task") == "external":
        if len(receipt["tasks"]) != 1:
            raise ValueError("External evaluations execute individually")
        args = [
            "olmo-eval",
            "run-external",
            "--model",
            checkpoint,
            "--eval",
            first["task"],
            "--provider",
            "vllm_server",
            "--base-url",
            "http://127.0.0.1:30000/v1",
            "--provider-kwarg",
            f"tokenizer={checkpoint}",
            "--provider-kwarg",
            "trust_remote_code=true",
            "--output-dir",
            str(output),
        ]
        for key, value in first["arguments"].items():
            args.extend(["--arg", f"{key}={json.dumps(value)}"])
        return args
    args = ["olmo-eval", "run", "--model", checkpoint, "--harness", first.get("harness_preset", "default")]
    for key, value in {
        "provider.kind": "sglang",
        "provider.tokenizer": checkpoint,
        "provider.trust_remote_code": True,
        "provider.kwargs.model_registration": "olmo_sglang:register",
        "provider.kwargs.tensor_parallel_size": receipt["evaluation"]["gpus"],
        "provider.kwargs.profile": "auto",
        # The provider recommends a sequence budget after checking actual engine
        # capacity. olmo-eval counts expanded samples (e.g. pass@32) against it.
        "batching.chunk_size": "auto",
        **{f"provider.kwargs.{key}": value for key, value in receipt["evaluation"]["provider_kwargs"].items()},
    }.items():
        args.extend(["-o", f"{key}={value if isinstance(value, str) else json.dumps(value)}"])
    for key, value in first.get("harness", {}).items():
        args.extend(["-o", f"{key}={value if isinstance(value, str) else json.dumps(value)}"])
    for task in receipt["tasks"]:
        args.extend(["--task", task["task"]])
        for key, value in task["generation"].items():
            args.extend(["-o", f"{key}={value if isinstance(value, str) else json.dumps(value)}"])
        for key, value in task["scoring"].items():
            args.extend(["-o", f"{key}={value if isinstance(value, str) else json.dumps(value)}"])
    return [*args, "--output-dir", str(output), "--save-predictions", "--save-requests"]


def scicode_server_options(provider_kwargs):
    """Translate serving aliases and reject options owned by the inline provider."""
    options = dict(provider_kwargs)
    unsupported = {"add_bos_token", "prompt_format", "force_download"}.intersection(options)
    if unsupported:
        raise ValueError(
            "SciCode's HTTP server does not accept native-provider-only options: "
            f"{', '.join(sorted(unsupported))}. Remove them from evaluation.provider_kwargs, "
            "or use a separate native-only evaluation configuration."
        )
    for alias, key in (("max_model_len", "context_length"), ("max_concurrency", "max_running_requests")):
        if alias in options:
            if key in options and options[key] != options[alias]:
                raise ValueError(f"SciCode serving options {alias} and {key} must agree")
            options[key] = options.pop(alias)
    return options


def scicode_checkpoint(checkpoint, output):
    """Normalize HF's disabled top-k sentinel for SGLang without changing the snapshot."""
    source = Path(checkpoint).resolve()
    config = source / "generation_config.json"
    if not config.is_file():
        return str(source)
    original = json.loads(config.read_text())
    if original.get("top_k") != 0:
        return str(source)
    # Keep weight symlinks outside the result tree: artifact collection copies
    # that tree with symlink dereferencing enabled.
    view = Path(tempfile.mkdtemp(prefix="miles-scicode-checkpoint-"))
    for path in source.iterdir():
        if path.name == "generation_config.json":
            continue
        target = view / path.name
        if target.is_symlink() and target.resolve() == path.resolve():
            continue
        target.symlink_to(path.resolve(), target_is_directory=path.is_dir())
    write_json(view / "generation_config.json", {**original, "top_k": -1})
    write_json(
        Path(output) / "scicode-sampling-normalization.json",
        {
            "checkpoint": str(source),
            "generation_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "original_top_k": 0,
            "normalized_top_k": -1,
        },
    )
    return str(view)


def serve(receipt, output):
    """Give SciCode's HTTP policy server the native evaluator's GPU/model profile."""
    options = scicode_server_options(receipt["evaluation"]["provider_kwargs"])
    selector = options.pop("profile", "auto")
    options.update(
        model_path=receipt["checkpoint"],
        served_model_name=receipt.get("served_model_name", receipt["checkpoint"]),
        tokenizer_path=receipt["checkpoint"],
        trust_remote_code=True,
        tp_size=receipt["evaluation"]["gpus"],
        host="127.0.0.1",
        port=30000,
    )
    profiles = importlib.import_module("olmo_eval.inference.providers.sglang_profiles")
    report = profiles.prepare_profile(selector, options, "olmo_sglang:register")
    options = {**report["defaults"], **options}
    write_json(Path(output) / "sglang-external-profile.json", {**report, "effective_options": options})
    print(f"SciCode SGLang profile: {report}", flush=True)
    importlib.import_module("sglang.srt.plugins").load_plugins()
    importlib.import_module("olmo_sglang").register()
    server_args = importlib.import_module("sglang.srt.server_args").ServerArgs(**options)
    try:
        importlib.import_module("sglang.launch_server").run_server(server_args)
    finally:
        importlib.import_module("sglang.srt.utils").kill_process_tree(os.getpid(), include_parent=False)


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
    paths = []
    external = False
    for root in result_roots(output):
        scicode = root / "scicode_results.json"
        if scicode.is_file():
            validate_scicode(scicode)
            external = True
        else:
            found = list(root.rglob("*-predictions.jsonl"))
            if not found:
                raise ValueError("No evaluation predictions found")
            paths.extend(found)
    if not paths and not external:
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


def validate_scicode(path):
    """SciCode can report success and zero scores after per-problem infrastructure errors."""
    result = json.loads(path.read_text())
    rows = result.get("predictions")
    if result.get("name") != "scicode" or result.get("success") is not True or result.get("error") or not rows:
        raise ValueError("Incomplete SciCode evaluation results")
    ids = set()
    total = 0
    for row in rows:
        count = row.get("total")
        problem = row.get("problem_id")
        if (
            not problem
            or problem in ids
            or row.get("error")
            or type(count) is not int
            or count < 1
            or len(row.get("step_results", [])) != count
            or len(row.get("step_codes", {})) != count
            or len(row.get("step_texts", {})) != count
        ):
            raise ValueError(f"Incomplete SciCode problem: {problem}")
        ids.add(problem)
        total += count
    metrics = result.get("metrics", {})
    if metrics.get("num_problems") != len(rows) or metrics.get("num_sub_steps") != total:
        raise ValueError("SciCode result counts disagree with predictions")


def scores(output):
    """Read the pinned olmo-eval metrics schema, without treating failures as zero."""
    files = [path for root in result_roots(output) for path in root.rglob("metrics.json")]
    if not files:
        raise ValueError("No olmo-eval metrics.json found")
    if any(
        (root / "scicode_results.json").is_file() or any(root.rglob("*-predictions.jsonl"))
        for root in result_roots(output)
    ):
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
    publication = Path(output) / "publication.json"
    previous = json.loads(publication.read_text()) if publication.is_file() else {}
    if wandb_run:
        tracking["entity"], tracking["project"], tracking["id"] = wandb_run.split("/")
    elif tracking["mode"] != "online":
        # An offline resume must not erase confirmation of an earlier manual upload.
        if previous.get("status") == "published":
            return
        write_json(
            Path(output) / "publication.json",
            {"status": "deferred", "reason": "Training offline or disabled; publish manually after sync"},
        )
        return
    if not all(tracking.get(key) for key in ("id", "entity", "project")):
        raise ValueError("Publishing requires the exact training entity/project/run ID")
    if (
        previous.get("status") == "published"
        and previous.get("update") == receipt["update"]
        and all(previous.get("run", {}).get(key) == tracking[key] for key in ("id", "entity", "project"))
    ):
        print(f"Reusing successful W&B publication: {output}", flush=True)
        return
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
        "evaluator": {
            key: config.get(key)
            for key in ("revision", "image", "gpus", "provider_kwargs", "external_server_args", "worker")
        },
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


def gpu_environments(receipt):
    """Expose the allocation to the native planner; isolate the external SciCode server."""
    policy_count = receipt["evaluation"]["gpus"]
    auxiliary_count = max(task.get("auxiliary_gpus", 0) for task in receipt["tasks"])
    total = policy_count + auxiliary_count
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    # This worker is copied alone into the evaluator image. Unlike the pinned
    # MILES adapter, evaluator providers accept GPU/MIG UUID masks. Keep that
    # capability while treating hidden/duplicate devices like the startup helper.
    devices = (
        [device.strip() for device in mask.split(",")] if mask is not None else [str(index) for index in range(total)]
    )
    if (
        len(devices) < total
        or len(set(devices)) != len(devices)
        or any(not device or device == "-1" for device in devices)
    ):
        raise ValueError(f"Evaluation needs {total} visible GPUs for policy and auxiliary providers")
    policy = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(devices[:policy_count])}
    external = receipt["tasks"][0].get("kind") == "external"
    visible = devices[policy_count:total] if external else devices[:total]
    evaluator = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(visible)}
    # The independent evaluator/judge environments must not import training's source overlays.
    evaluator.pop("PYTHONPATH", None)
    return policy, evaluator


def print_failure_log(path):
    """Expose bounded evaluator diagnostics in Beaker without masking the failure."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 16384))
            tail = "\n".join(stream.read().decode("utf-8", errors="replace").splitlines()[-80:])
        print(f"olmo-eval failed; log tail from {path}:\n{tail}", file=sys.stderr, flush=True)
    except OSError as error:
        print(f"Could not read evaluator log {path}: {error}", file=sys.stderr, flush=True)


def run_tasks(receipt, output):
    completed = []
    failures = []
    for task in receipt["tasks"]:
        directory, attempt = saved_task(output, task)
        if attempt is None:
            # olmo-eval has no response-level resume contract here. Keep partial
            # attempts intact and rerun only this unfinished task in a fresh path.
            attempt = directory / "attempts" / uuid.uuid4().hex
            attempt.mkdir(parents=True)
            args = command({**receipt, "tasks": [task]}, attempt)
            write_json(attempt / "command.json", args)
            log_path = attempt / "olmo-eval.log"
            try:
                with log_path.open("w") as log:
                    subprocess.run(
                        args, stdout=log, stderr=subprocess.STDOUT, check=True, env=gpu_environments(receipt)[1]
                    )
                values = completed_scores(attempt, [task])
            except (subprocess.CalledProcessError, ValueError) as error:
                print_failure_log(log_path)
                failures.append((task["task"], error))
                continue
            write_json(
                directory / "complete.json",
                {"task": task, "attempt": str(attempt.relative_to(directory)), "scores": values},
            )
        else:
            print(f"Reusing completed evaluation task: {task['task']}", flush=True)
        completed.append(str(attempt.relative_to(output)))
    write_json(output / "task-results.json", completed)
    write_json(output / "task-failures.json", [{"task": task, "error": str(error)} for task, error in failures])
    if failures:
        raise failures[0][1]


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
        if (
            any(task.get("kind") == "external" or task.get("harness_preset") == "codex_python" for task in pending)
            and shutil.which("podman") is None
        ):
            raise RuntimeError("Code evaluation requires a sandbox-enabled evaluator image with Podman")
        for task in pending:
            if task.get("kind") == "external" and not Path(task["arguments"]["h5py_host_path"]).is_file():
                raise ValueError("SciCode reference data is missing: arguments.h5py_host_path")
        revision = subprocess.check_output(
            ["git", "-C", "/opt/olmo-eval", "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
        if revision != receipt["evaluation"]["revision"]:
            raise ValueError("Evaluator revision does not match the pinned receipt")
        checkpoint = receipt["checkpoint"]
        for name in ("config.json", "tokenizer_config.json"):
            if not (Path(checkpoint) / name).is_file():
                raise ValueError(f"Incomplete HF snapshot: missing {name}")
        # SciCode currently requires VLLMServerProvider; all regular tasks use the inline engine.
        if any(task.get("kind") == "external" for task in pending):
            checkpoint = scicode_checkpoint(checkpoint, output)
            env = {**gpu_environments(receipt)[0], "SGLANG_EXTERNAL_MODEL_PACKAGE": "olmo_sglang.models"}
            if not receipt["evaluation"]["external_server_args"]:
                server_receipt = output / "server-receipt.json"
                write_json(
                    server_receipt, {**receipt, "checkpoint": checkpoint, "served_model_name": receipt["checkpoint"]}
                )
                server_args = [sys.executable, __file__, "serve", str(server_receipt), "--results", str(output)]
            else:
                server_args = [
                    sys.executable,
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    checkpoint,
                    "--served-model-name",
                    receipt["checkpoint"],
                    "--trust-remote-code",
                    "--tokenizer-path",
                    checkpoint,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "30000",
                    "--tp-size",
                    str(receipt["evaluation"]["gpus"]),
                    *receipt["evaluation"]["external_server_args"],
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
    parser.add_argument("command", choices=("run", "publish", "serve"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--wandb-run", help="entity/project/run_id, after offline training has been synced")
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if args.command == "run":
        run_evaluation(receipt)
    elif args.command == "serve":
        serve(receipt, args.results or result_directory(receipt))
    elif not try_publish(receipt, args.results or result_directory(receipt), wandb_run=args.wandb_run):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
