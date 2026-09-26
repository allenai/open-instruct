"""Record every scored training group, kept or filtered, with identity, validity and disposition.

A run's rollouts are its most expensive product. The trainer consumes only the
groups that pass the online filter and stay within the policy-lag limit; these
records keep the outcome of every scored group so later runs can select prompts,
and analyses can price the work that filtering discarded. Records are
append-only JSONL in a shared store:

    <root>/<lineage>/<run name>-<run id>/manifest-<attempt>.json
    <root>/<lineage>/<run name>-<run id>/records-<attempt>.jsonl

``lineage`` digests the starting checkpoint's recorded file inventory: path,
sizes, modification times and JSON hashes. It is not a digest of the weights.
Each response separately records the policy versions that generated it.

Recording never changes admission, filtering or training. Rows are queued to a
bounded background writer; a full queue drops rows and counts them rather than
blocking the rollout path.
"""

import atexit
import hashlib
import json
import os
import queue
import socket
import threading
import time
import uuid
from pathlib import Path

from open_instruct import logger_utils
from open_instruct.miles.execution import workflow
from open_instruct.miles.publication import policy_versions

logger = logger_utils.setup_logger(__name__)

SCHEMA_VERSION = 1
MARKER = "workflow-model.json"
QUEUE_LIMIT = 4096
CLOSE_TIMEOUT_SECONDS = 30.0
# Verifier outcomes that make a reward valid evidence. "completed" is the adapter's
# own status for verifiers that return normally without reporting diagnostics.
VALID_STATUSES = frozenset({"ok", "completed"})
PROTOCOL_FIELDS = (
    "rollout_temperature",
    "rollout_top_p",
    "rollout_top_k",
    "rollout_max_response_len",
    "rollout_max_context_len",
    "rollout_max_prompt_len",
    "n_samples_per_prompt",
    "rollout_stop",
    "rollout_stop_token_ids",
    "sglang_enable_deterministic_inference",
)
TEMPLATE_FILES = ("tokenizer_config.json", "chat_template.jinja", "generation_config.json")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _file_sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError):
        return None


def task_identity(metadata, prompt):
    """Identify the task by its complete rendered input and verifier targets.

    The rendered prompt contains every system and conversation turn, so tasks that
    share only a final user message stay distinct. It is specific to the chat
    template recorded in the manifest's protocol. ``query_sha256`` hashes only the
    final user message and is a grouping hint, never an identity.
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    verifiers = [
        {"name": spec.get("name"), "target": spec.get("target")}
        for spec in metadata.get("verifiers") or []
        if isinstance(spec, dict)
    ]
    query = metadata.get("query")
    return {
        "task_key": sha256({"basis": "rendered", "prompt": prompt, "verifiers": verifiers}),
        "task_key_basis": "rendered",
        "query_sha256": sha256(query) if isinstance(query, str) else None,
    }


def lineage_identity(checkpoint):
    """Digest the starting checkpoint's file inventory, recorded at workflow preparation.

    Every run prepared from the same source shares this digest. Without the marker,
    the served directory's own inventory is used and ``basis`` says so.
    """
    path = Path(checkpoint)
    marker = path / MARKER
    if marker.is_file():
        identity = json.loads(marker.read_text())["identity"]["source"]
        basis = "source_inventory"
    else:
        identity = workflow.model_identity(path)
        basis = "served_inventory"
    return {
        "inventory_sha256": sha256(identity),
        "basis": basis,
        "path": identity["path"],
        "served_path": str(path),
        "weights_hashed": False,
    }


def protocol_identity(args):
    """Settings that change what a sampled outcome estimates; pool only equal protocols."""
    checkpoint = Path(getattr(args, "hf_checkpoint", "") or ".")
    protocol = {name: getattr(args, name, None) for name in PROTOCOL_FIELDS}
    protocol["template_files_sha256"] = {name: _file_sha256(checkpoint / name) for name in TEMPLATE_FILES}
    core = args.olmo_core
    protocol["reward_config_sha256"] = _file_sha256(core.reward_config)
    protocol["filter_zero_std_groups"] = core.filter_zero_std_groups
    return protocol


def include_responses(mode, rate, observation):
    """Sample whole groups deterministically so siblings stay comparable."""
    if mode == "all":
        return True
    if mode != "sample":
        return False
    return int(hashlib.sha256(observation.encode()).hexdigest()[:8], 16) / 2**32 < rate


def validity(metadata):
    """One state per expected verifier; missing diagnostics are unknown, never valid or failed."""
    metadata = metadata or {}
    diagnostics = metadata.get("verifier_diagnostics") or {}
    expected = [spec.get("name") for spec in metadata.get("verifiers") or [] if isinstance(spec, dict)]
    components = {}
    for name in expected:
        record = diagnostics.get(name)
        status = record.get("status") if isinstance(record, dict) else None
        components[name] = status if isinstance(status, str) and status else "unknown"
    states = set(components.values())
    valid = None if not components or "unknown" in states else states <= VALID_STATUSES
    return {"components": components, "valid": valid}


def _version_order(value):
    return (0, int(value)) if value.isdigit() else (1, value)


def policy_scope(versions, fresh_run):
    """Separate exact starting-checkpoint samples from later or mixed-policy samples.

    Weight versions count completed optimizer steps, and a fresh run publishes the
    starting checkpoint as version 0. Later versions are specific to this run's
    trajectory and must not be equated with the same version of another run.
    """
    if not versions:
        return "unknown"
    if len(versions) > 1:
        return "mixed"
    if fresh_run and versions[0] == "0":
        return "start_checkpoint"
    return "run_version"


def response_record(sample, text, fresh_run):
    metadata = sample.metadata or {}
    versions = sorted(
        {str(value) for value in policy_versions.versions(sample.weight_versions, allow_empty=True)}
        if sample.weight_versions
        else set(),
        key=_version_order,
    )
    status = getattr(sample.status, "value", str(sample.status))
    record = {
        "index": sample.index,
        "reward": sample.reward,
        "reward_components": [
            {key: item.get(key) for key in ("name", "score", "weight")}
            for item in metadata.get("reward_components") or []
        ],
        "validity": validity(metadata),
        "response_tokens": sample.response_length,
        "status": status,
        "truncated": status == "truncated",
        "policy_versions": versions,
        "policy_scope": policy_scope(versions, fresh_run),
    }
    if text:
        record["response"] = sample.response
    return record


class Recorder:
    """Queue group and disposition rows for a background writer owned by the rollout process."""

    def __init__(self, args):
        core = args.olmo_core
        self.root = Path(core.records_root)
        self.responses = core.records_responses
        self.rate = core.records_response_sample_rate
        self.attempt = uuid.uuid4().hex[:12]
        self.run_name = getattr(args, "wandb_run_name", None) or "run"
        self.run_id = os.environ.get("BEAKER_WORKLOAD_ID") or os.environ.get("BEAKER_EXPERIMENT_ID") or self.attempt
        self.fresh_run = getattr(args, "start_rollout_id", 0) == 0
        self.counts = {"queued": 0, "written": 0, "dropped": 0, "failed": 0}
        self._queue = queue.Queue(maxsize=QUEUE_LIMIT)
        self._lock = threading.Lock()
        self.unavailable = False
        try:
            self.lineage = lineage_identity(args.hf_checkpoint)
            self.protocol = protocol_identity(args)
        except Exception:
            self.unavailable = True
            logger.exception("Inference records disabled: checkpoint lineage or protocol unavailable")
            return
        self.protocol_sha256 = sha256(self.protocol)
        self.directory = self.root / self.lineage["inventory_sha256"][:16] / f"{self.run_name}-{self.run_id}"
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "source": "train",
            "run": {
                "name": self.run_name,
                "id": self.run_id,
                "attempt": self.attempt,
                "start_rollout_id": getattr(args, "start_rollout_id", None),
                "load": getattr(args, "load", None),
                "fresh": self.fresh_run,
                "beaker_experiment_id": os.environ.get("BEAKER_EXPERIMENT_ID"),
                "host": os.environ.get("BEAKER_NODE_HOSTNAME") or socket.gethostname(),
            },
            "started_unix": time.time(),
            "lineage": self.lineage,
            "protocol": self.protocol,
            "protocol_sha256": self.protocol_sha256,
            "rollout_seed": getattr(args, "rollout_seed", None),
            "prompt_data": getattr(args, "prompt_data", None),
            "records": {"responses": self.responses, "response_sample_rate": self.rate, "queue_limit": QUEUE_LIMIT},
            # Records from a run with prompt selection describe a filtered prompt stream.
            "selection_sha256": getattr(core, "selection_sha256", None),
        }
        self._writer = threading.Thread(target=self._write, name="inference-records", daemon=True)
        self._writer.start()
        atexit.register(self.close)

    def _count(self, key, amount=1):
        with self._lock:
            self.counts[key] += amount

    def _enqueue(self, row):
        if self.unavailable:
            self._count("dropped")
            return
        try:
            self._queue.put_nowait(row)
            self._count("queued")
        except queue.Full:
            self._count("dropped")

    def record_group(self, samples, *, decision, reason=None):
        """Record one scored group and stamp its observation ID for later disposition events."""
        if self.unavailable:
            self._count("dropped")
            return
        observation = uuid.uuid4().hex
        try:
            first = samples[0]
            metadata = first.metadata or {}
            for sample in samples:
                sample.metadata = {**(sample.metadata or {}), "inference_record_id": observation}
            text = include_responses(self.responses, self.rate, observation)
            identity = task_identity(metadata, first.prompt)
            token_sha256 = metadata.get("run_prompt_token_ids_sha256")
            row = {
                "schema_version": SCHEMA_VERSION,
                "kind": "group",
                "observation_id": observation,
                "recorded_unix": time.time(),
                **identity,
                # Exact input and task under one protocol: the pooling unit for evidence.
                "input_key": sha256(
                    {"task": identity["task_key"], "tokens": token_sha256, "protocol": self.protocol_sha256}
                ),
                "prompt_token_sha256": token_sha256,
                "prepared_sample_id": metadata.get("prepared_sample_id"),
                "source_dataset": metadata.get("source_dataset"),
                "source_row": metadata.get("source_row"),
                "verifiers": [spec.get("name") for spec in metadata.get("verifiers") or [] if isinstance(spec, dict)],
                "group_index": first.group_index,
                "rollout_id": first.rollout_id,
                "group_attempt": (metadata.get("sibling_timing") or {}).get("group_attempt"),
                "group_size": len(samples),
                "filter_decision": decision,
                "filter_reason": reason,
                "responses_included": text,
                "responses": [response_record(sample, text, self.fresh_run) for sample in samples],
            }
        except Exception:
            self._count("failed")
            logger.exception("Inference record unavailable for group %s", getattr(samples[0], "group_index", None))
            return
        self._enqueue(row)

    def record_disposition(self, samples, *, accepted, staleness):
        """A passed group left the completed queue: consumed by training, or expired past the lag limit."""
        observation = ((samples[0].metadata or {}) if samples else {}).get("inference_record_id")
        if observation is None:
            return
        self._enqueue(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "disposition",
                "observation_id": observation,
                "recorded_unix": time.time(),
                "disposition": "consumed" if accepted else "expired",
                "staleness": staleness,
            }
        )

    def _write(self):
        stream = None
        while True:
            row = self._queue.get()
            if row is None:
                break
            if self.unavailable:
                self._count("dropped")
                continue
            try:
                if stream is None:
                    self.directory.mkdir(parents=True, exist_ok=True)
                    manifest = self.directory / f"manifest-{self.attempt}.json"
                    manifest.write_text(json.dumps(self.manifest, indent=2, default=str) + "\n")
                    stream = (self.directory / f"records-{self.attempt}.jsonl").open("a")
                    logger.info("Recording inference outcomes to %s", self.directory)
                stream.write(json.dumps(row, allow_nan=False, default=str) + "\n")
                stream.flush()
                self._count("written")
            except Exception:
                self._count("failed")
                if stream is None and not self.unavailable:
                    # Report once; retrying an unavailable store for every row would flood the log.
                    self.unavailable = True
                    logger.exception("Inference records disabled for this process: store %s unavailable", self.root)
                elif stream is not None:
                    logger.exception("Inference record write failed")
        if stream is not None:
            stream.close()

    def close(self, timeout=CLOSE_TIMEOUT_SECONDS):
        """Flush queued rows for a bounded time; rows still pending afterwards are reported, not waited on."""
        writer = getattr(self, "_writer", None)
        if writer is None or not writer.is_alive():
            return
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            logger.warning("Inference records: writer backlog did not drain; %d rows pending", self._queue.qsize())
            return
        writer.join(timeout)
        if writer.is_alive():
            logger.warning("Inference records: writer did not finish within %.0f s", timeout)

    def metrics(self):
        with self._lock:
            counts = dict(self.counts)
        return {
            **{f"rollout/records/{key}_total": value for key, value in counts.items()},
            "rollout/records/pending": self._queue.qsize(),
        }
