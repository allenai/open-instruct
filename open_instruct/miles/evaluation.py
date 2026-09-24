"""Best-effort evaluation planning and single-coordinator Beaker submission.

This module is CPU-only. Snapshot collectives live in the actor; nothing here
owns rollout engines, checkpoint retention, retries, or evaluator lifetimes.
"""

import argparse
import base64
import hashlib
import json
import re
import shlex
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from open_instruct import logger_utils
from open_instruct.miles import state
from open_instruct.miles.errors import InputError

logger = logger_utils.setup_logger(__name__)


def parse(value, launch):
    value = dict(value)
    allowed = {
        "mode",
        "interval",
        "initial",
        "final",
        "tasks",
        "image",
        "revision",
        "gpus",
        "cluster",
        "workspace",
        "budget",
        "secrets",
        "submit_timeout",
        "timeout",
        "server_args",
    }
    if set(value) - allowed:
        raise InputError(f"Unknown evaluation fields: {sorted(set(value) - allowed)}")
    mode = value.setdefault("mode", "shared")
    if mode not in ("shared", "background"):
        raise InputError("evaluation.mode must be shared or background")
    if mode == "shared":
        if set(value) != {"mode"}:
            raise InputError(
                "evaluation settings require mode=background; shared evaluation uses existing data/miles fields"
            )
        return value
    defaults = dict(
        interval=20, initial=True, final=True, gpus=1, submit_timeout=30, timeout="3h", secrets={}, server_args=[]
    )
    for key, default in defaults.items():
        value.setdefault(key, default)
    for key in ("cluster", "workspace", "budget"):
        value.setdefault(key, launch[key])
    for key in ("interval", "gpus", "submit_timeout"):
        if type(value[key]) is not int or value[key] < 1:
            raise InputError(f"evaluation.{key} must be a positive integer")
    for key in ("initial", "final"):
        if type(value[key]) is not bool:
            raise InputError(f"evaluation.{key} must be boolean")
    if not isinstance(value.get("image"), str) or not re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", value["image"]):
        raise InputError("evaluation.image must be an immutable Beaker image ID")
    if not isinstance(value.get("revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", value["revision"]):
        raise InputError("evaluation.revision must pin the full olmo-eval Git commit")
    for key in ("cluster", "workspace", "budget", "timeout"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise InputError(f"evaluation.{key} must be a nonempty string")
    secrets = value["secrets"]
    if not isinstance(secrets, dict) or any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or not isinstance(secret, str) or not secret
        for key, secret in secrets.items()
    ):
        raise InputError("evaluation.secrets must map environment names to Beaker secret references")
    value["secrets"] = {
        key: secret for key, secret in launch["secrets"].items() if key in {"WANDB_API_KEY", "HF_TOKEN"}
    } | secrets
    if not isinstance(value["server_args"], list) or any(not isinstance(arg, str) for arg in value["server_args"]):
        raise InputError("evaluation.server_args must be a list of SGLang arguments")
    reserved = {"--model", "--model-path", "--tokenizer-path", "--port", "--host", "--tp", "--tp-size"}
    if any(arg.split("=")[0] in reserved for arg in value["server_args"]):
        raise InputError("evaluation.server_args cannot override snapshot, tokenizer, port or GPU allocation")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise InputError("evaluation.tasks must be a nonempty list of olmo-eval tasks")
    names = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) - {"task", "interval", "generation", "scoring"}:
            raise InputError("evaluation.tasks accept task, interval, generation and scoring")
        name = task.get("task")
        if not isinstance(name, str) or not name or name in names:
            raise InputError("evaluation task names must be nonempty and unique")
        names.add(name)
        task.setdefault("interval", value["interval"])
        if type(task["interval"]) is not int or task["interval"] < 1:
            raise InputError("evaluation task interval must be a positive integer")
        for key in ("generation", "scoring"):
            task.setdefault(key, {})
            if not isinstance(task[key], dict):
                raise InputError(f"evaluation task {key} must be an override table")
        # No credential values are accepted in overrides; use evaluator secrets.
        encoded = json.dumps(task).lower()
        if any(word in encoded for word in ("api_key", "password", "access_token", "api-key")):
            raise InputError("Use evaluation.secrets for credentials, not task overrides")
    return value


def groups(config, update, total):
    """Merge coincident initial/periodic/final milestones, grouped by harness settings."""
    selected = {}
    for task in config["tasks"]:
        due = config["initial"] if update == 0 else update % task["interval"] == 0
        due = due or (update == total and config["final"])
        if due:
            key = json.dumps(task["generation"], sort_keys=True)
            selected.setdefault(key, []).append(task)
    return list(selected.values())


def total_updates(options):
    return (
        options["num_rollout"]
        * options["rollout_batch_size"]
        * options["n_samples_per_prompt"]
        // options["global_batch_size"]
    )


def snapshot(root, update):
    return Path(root) / "eval-snapshots" / f"update-{update:08d}" / "hf"


def plan(spec, options):
    config = spec.evaluation
    if config["mode"] != "background":
        return config
    total = total_updates(options)
    updates = {0} if config["initial"] else set()
    if config["final"]:
        updates.add(total)
    for task in config["tasks"]:
        updates.update(range(task["interval"], total + 1, task["interval"]))
    return {
        **config,
        "mounts": spec.launch["weka_mounts"],
        "milestones": [
            {
                "update": update,
                "groups": groups(config, update, total),
                "snapshot": str(snapshot(spec.output["root"], update)),
            }
            for update in sorted(updates)
        ],
        "initial_checkpoint_reuse": spec.conversion["hf_output"],
        "retention": "manual cleanup only",
    }


def runtime(spec, options):
    return {
        **spec.evaluation,
        "root": spec.output["root"],
        "name": spec.name,
        "mounts": spec.launch["weka_mounts"],
        "total_updates": total_updates(options),
    }


def specification(receipt):
    config = receipt["evaluation"]
    # Carry the committed runner to the independent evaluator image. No source
    # checkout, dependency resolution or moving branch is used at job startup.
    runner = Path(__file__).with_name("evaluation_runner.py").read_bytes()
    if hashlib.sha256(runner).hexdigest() != receipt["runner_sha256"]:
        raise InputError("Evaluator runner differs from the recorded source; use its committed checkout")
    payload = base64.b64encode(json.dumps(receipt).encode()).decode()
    source = base64.b64encode(runner).decode()
    setup = f"import base64,pathlib; pathlib.Path('/tmp/evaluate.py').write_bytes(base64.b64decode({source!r})); pathlib.Path('/tmp/evaluation.json').write_bytes(base64.b64decode({payload!r}))"
    return {
        "version": "v2",
        "budget": config["budget"],
        "description": f"MILES background evaluation {receipt['training']['name']} update {receipt['update']}",
        "tasks": [
            {
                "name": "evaluation",
                "image": {"beaker": config["image"]},
                "command": ["bash", "-c"],
                "arguments": [
                    f"set -euo pipefail\npython -c {shlex.quote(setup)}\npython /tmp/evaluate.py run /tmp/evaluation.json"
                ],
                "datasets": [{"mountPath": m["mount_path"], "source": {"weka": m["weka"]}} for m in config["mounts"]],
                "result": {"path": "/output"},
                "resources": {"gpuCount": config["gpus"], "sharedMemory": "32 GiB"},
                "constraints": {"cluster": [config["cluster"]]},
                "context": {"priority": "normal", "autoResume": False},
                "timeout": config["timeout"],
                "envVars": [{"name": name, "secret": secret} for name, secret in config["secrets"].items()],
            }
        ],
    }


def submit(receipt, path):
    """One attempt with a hard process deadline. Timeout means ambiguous submission."""
    spec_path = path.with_suffix(".beaker.json")
    try:
        if receipt["update"] and not (Path(receipt["checkpoint"]) / ".complete").is_file():
            raise ValueError("Snapshot completion marker missing; refusing submission")
        state.atomic_json(spec_path, specification(receipt))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "open_instruct.miles.evaluation_submit",
                str(spec_path),
                receipt["evaluation"]["workspace"],
                f"miles-eval-{hashlib.sha256(receipt['evaluation']['root'].encode()).hexdigest()[:10]}-{receipt['update']}-{receipt['group_id']}",
                str(receipt["evaluation"]["submit_timeout"]),
            ],
            capture_output=True,
            text=True,
            timeout=receipt["evaluation"]["submit_timeout"],
            check=True,
        )
        response = json.loads(result.stdout)
        experiment = response[0] if isinstance(response, list) else response
        receipt.update(status="submitted", experiment_id=experiment["id"])
    except Exception as error:
        # Do not persist child stdout/stderr: tools may echo credential values.
        receipt.update(
            status="submission_failed",
            error=type(error).__name__,
            diagnostic="Submission failed or timed out; inspect Beaker before manual resubmission. No automatic retry.",
        )
        if isinstance(error, subprocess.CalledProcessError):
            receipt["exit_code"] = error.returncode
            try:
                # Only the controlled child emits this redacted JSON diagnostic.
                detail = json.loads((error.stderr or "").splitlines()[-1])
                if isinstance(detail, dict) and set(detail) == {"error", "message"}:
                    receipt["submission_error"] = detail
            except (ValueError, IndexError):
                pass
        logger.warning(
            "BACKGROUND EVALUATION GAP at update %s: %s; receipt %s",
            receipt["update"],
            receipt.get("submission_error", type(error).__name__),
            path,
        )
    state.atomic_json(path, receipt)


class Coordinator:
    """Driver-only submitter, with one daemon worker and no pending queue or join."""

    def __init__(self, config, training):
        self.config = config
        self.training = training
        self.worker = None

    def dispatch(self, update, checkpoint, *, final=False):
        try:
            self._dispatch(update, checkpoint, final=final)
        except Exception as error:
            # Local receipt/submission preparation is independent of trainer health.
            logger.warning(
                "BACKGROUND EVALUATION GAP at update %s: cannot record/submit (%s)", update, type(error).__name__
            )

    def _dispatch(self, update, checkpoint, *, final=False):
        if update == 0:
            state.atomic_json(
                snapshot(self.config["root"], 0).parent / "reference.json",
                {"checkpoint": str(checkpoint), "update": 0, "reuse_initial_hf": True},
            )
        receipts = []
        for tasks in groups(self.config, update, update if final else self.config["total_updates"]):
            key = hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()[:16]
            path = Path(self.config["root"]) / "evaluation" / f"update-{update:08d}-{key}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive claim also protects against a duplicate coordinator call.
            try:
                with path.open("x") as stream:
                    json.dump({"status": "claimed", "update": update}, stream)
            except FileExistsError:
                continue  # Including ambiguous/failed attempts: never retry automatically.
            receipt = dict(
                schema_version=1,
                update=update,
                checkpoint=str(checkpoint),
                tasks=tasks,
                group_id=key,
                evaluation=self.config,
                training=self.training,
                status="pending",
                runner_sha256=hashlib.sha256(
                    Path(__file__).with_name("evaluation_runner.py").read_bytes()
                ).hexdigest(),
            )
            if self.worker is not None and self.worker.is_alive():
                receipt.update(status="skipped_busy", diagnostic="Submission worker busy; milestone dropped")
                logger.warning("BACKGROUND EVALUATION GAP at update %s: submission worker busy", update)
            state.atomic_json(path, receipt)
            if receipt["status"] == "pending":
                receipts.append((receipt, path))
        if receipts:
            self.worker = threading.Thread(
                target=self._submit, args=(receipts,), daemon=True, name="miles-evaluation-submit"
            )
            self.worker.start()

    @staticmethod
    def _submit(receipts):
        for receipt, path in receipts:
            try:
                submit(receipt, path)
            except Exception as error:
                logger.warning("BACKGROUND EVALUATION GAP: receipt %s (%s)", path, type(error).__name__)


def main():
    parser = argparse.ArgumentParser(description="Manually resubmit one evaluation receipt; no automatic retries")
    parser.add_argument("command", choices=("resubmit",))
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    # Preserve the old receipt/results and make the explicit new attempt reviewable.
    receipt.update(status="pending", group_id=uuid.uuid4().hex[:16], resubmission_of=str(args.receipt))
    receipt.pop("experiment_id", None)
    receipt.pop("error", None)
    path = args.receipt.with_name(f"{args.receipt.stem}-manual-{receipt['group_id']}.json")
    state.atomic_json(path, receipt)
    submit(receipt, path)
    print(path)
    if receipt["status"] != "submitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
