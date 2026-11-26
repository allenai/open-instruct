"""Durable handoff from the trainer to a separate Beaker export and evaluation job."""

import fcntl
import hashlib
import importlib
import json
import os
import shlex
import sys
import threading
import time
from pathlib import Path

from open_instruct import logger_utils
from open_instruct.miles.evaluation import evaluation
from open_instruct.miles.infrastructure.artifacts import atomic_json

logger = logger_utils.setup_logger(__name__)


def pin_checkpoint(save, rollout, update):
    """Retain the eval model across native retention without copying tensor data."""
    source = Path(save) / "core" / f"rollout_{rollout:07d}"
    target = Path(save) / "evaluation-native" / f"update-{update:08d}"
    manifest = json.loads((source / "complete.json").read_text())
    if manifest["clock"]["completed_steps"] != update:
        raise ValueError(f"Checkpoint {source} did not commit update {update}")
    model = source / "model"
    if not (model / ".metadata").is_file():
        raise FileNotFoundError(model / ".metadata")
    for file in sorted(model.rglob("*")):
        if not file.is_file():
            continue
        link = target / "model" / file.relative_to(model)
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(file, link)
        except FileExistsError:
            if not link.samefile(file):
                raise ValueError(f"Different checkpoint already pinned at {link}") from None
    atomic_json(target / "complete.json", manifest)
    return target


def specification(request, path):
    config = request["evaluation"]
    worker = config["export_worker"]
    return {
        "version": "v2",
        "description": f"MILES checkpoint export and evaluation at update {request['update']}",
        "budget": worker["budget"],
        "tasks": [
            {
                "name": "checkpoint-evaluation",
                "image": {"beaker": worker["image"]},
                "command": ["bash", "-c"],
                "arguments": [f"cd /opt/core-rl && exec python -m {__name__} {shlex.quote(str(path))}"],
                "datasets": [{"mountPath": m["mount_path"], "source": {"weka": m["weka"]}} for m in config["mounts"]],
                "resources": {"cpuCount": 8, "memory": "28 GiB"},
                "constraints": {"cluster": [worker["cluster"]]},
                "context": {"priority": "high", "autoResume": True},
                "envVars": [
                    {"name": "CUDA_VISIBLE_DEVICES", "value": ""},
                    {"name": "OMP_NUM_THREADS", "value": "8"},
                    {"name": "BEAKER_TOKEN", "secret": worker["token_secret"]},
                ],
                "timeout": "4h",
                "result": {"path": "/output"},
            }
        ],
    }


class Coordinator:
    """Persist requests in the driver, then submit them with bounded background API calls."""

    def __init__(self, config, training):
        self.config = config
        self.training = training
        self.root = Path(config["root"]) / "evaluation" / "checkpoint-requests"
        self.thread = None
        self.stop = threading.Event()

    def enqueue(self, update, tokenizer, *, save=None, rollout=None, final=False):
        try:
            self.record(update, tokenizer, save=save, rollout=rollout, final=final)
        except (OSError, ValueError, KeyError) as error:
            logger.error(
                "Could not record checkpoint evaluation request at update %d (%s)", update, type(error).__name__
            )
        self.kick()

    def recover(self, save, tokenizer, per_rollout):
        for path in sorted((Path(save) / "core").glob("rollout_*/complete.json")):
            rollout = int(path.parent.name.removeprefix("rollout_"))
            update = (rollout + 1) * per_rollout
            if evaluation.groups(self.config, update, self.config["total_updates"]):
                self.enqueue(update, tokenizer, save=save, rollout=rollout)
        self.kick()

    def record(self, update, tokenizer, *, save=None, rollout=None, final=False):
        path = self.root / f"update-{update:08d}.json"
        task_groups = evaluation.groups(self.config, update, update if final else self.config["total_updates"])
        if path.exists():
            if not final:
                return
            # Finite data can exhaust after a periodic group has already been
            # recorded at this update. Submit only the remaining final tasks.
            previous = json.loads(path.read_text())
            submitted = [task for group in previous["groups"] for task in group]
            task_groups = [[task for task in group if task not in submitted] for group in task_groups]
            task_groups = [group for group in task_groups if group]
            path = path.with_stem(path.stem + "-final")
        if not task_groups or path.exists():
            return
        checkpoint = (
            str(evaluation.snapshot(self.config["root"], update).with_name("checkpoint-hf"))
            if update
            else str(tokenizer)
        )
        request = {
            "update": update,
            "checkpoint": checkpoint,
            "native": str(pin_checkpoint(save, rollout, update)) if update else None,
            "tokenizer": str(tokenizer),
            "groups": task_groups,
            "evaluation": self.config,
            "training": self.training,
        }
        atomic_json(path, request)

    def kick(self):
        if self.thread is None or not self.thread.is_alive():
            self.stop.clear()
            self.thread = threading.Thread(
                target=self._submit_pending, name="miles-checkpoint-evaluation", daemon=True
            )
            self.thread.start()

    def _submit_pending(self):
        pending = sorted([*self.root.glob("update-????????.json"), *self.root.glob("update-????????-final.json")])
        for path in pending:
            if self.stop.is_set():
                return
            submitted = path.with_suffix(".submitted.json")
            if submitted.is_file():
                continue
            try:
                request = json.loads(path.read_text())
                worker = request["evaluation"]["export_worker"]
                document = path.with_suffix(".beaker.json")
                atomic_json(document, specification(request, path))
                root_id = hashlib.sha256(self.config["root"].encode()).hexdigest()[:12]
                experiment = evaluation.create_experiment(
                    document,
                    worker["workspace"],
                    f"miles-checkpoint-{root_id}-{path.stem}",
                    self.config["submit_timeout"],
                    reconcile=True,
                )
                atomic_json(submitted, experiment)
            except Exception as error:
                logger.warning("Checkpoint evaluation request %s remains pending (%s)", path, type(error).__name__)

    def drain(self):
        self.kick()
        deadline = time.monotonic() + self.config["submit_timeout"] + 1
        self.thread.join(max(0, deadline - time.monotonic()))
        if not self.thread.is_alive():
            self.kick()  # Include a final request recorded during the previous scan.
            self.thread.join(max(0, deadline - time.monotonic()))
        self.stop.set()


def run(path):
    request = json.loads(path.read_text())
    update = request["update"]
    if request["native"]:
        exporter = importlib.import_module("open_instruct.miles.evaluation.checkpoint_export")
        output = Path(request["checkpoint"])
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".export.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            for attempt in range(3):
                try:
                    exporter.export(Path(request["native"]), Path(request["tokenizer"]), output, update)
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    logger.exception("HF export I/O transient at update %d; retrying verified shards", update)
                    time.sleep(10 * (attempt + 1))
    for attempt in range(3):
        failed = submit_groups(request, path)
        if not failed:
            return
        if attempt < 2:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Pending evaluation groups {failed}; rerun checkpoint evaluation with {path}")


def submit_groups(request, path):
    config, update = request["evaluation"], request["update"]
    failed = []
    for tasks in request["groups"]:
        group = hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()[:16]
        receipt_path = path.parent.parent / f"update-{update:08d}-{group}.json"
        if receipt_path.is_file():
            previous = json.loads(receipt_path.read_text())
            if previous.get("status") == "submitted":
                continue
        receipt = {
            "schema_version": 1,
            "update": update,
            "checkpoint": request["checkpoint"],
            "tasks": tasks,
            "group_id": group,
            "evaluation": config,
            "training": request["training"],
            "status": "pending",
            "runner_sha256": hashlib.sha256(
                Path(evaluation.__file__).with_name("evaluation_runner.py").read_bytes()
            ).hexdigest(),
        }
        evaluation.submit(receipt, receipt_path, reconcile=True)
        if receipt["status"] != "submitted":
            failed.append(group)
    return failed


if __name__ == "__main__":
    run(Path(sys.argv[1]))
