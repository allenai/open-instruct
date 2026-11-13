"""Connect MILES inference records to the model source prepared by this workflow.
The recorder receives a stable identity derived from the starting checkpoint's
file inventory, so separate runs prepared from the same source can share evidence
for prompt selection. The identity records its basis explicitly: it describes an
inventory rather than a hash of every weight tensor.
"""

import copy
import json
import weakref
from pathlib import Path

from miles.utils import inference_records

from open_instruct import logger_utils
from open_instruct.miles.datasets import replay_evidence
from open_instruct.miles.execution import workflow
from open_instruct.miles.rewards import grading

logger = logger_utils.setup_logger(__name__)

MARKER = "workflow-model.json"
# Runtime argument copies are shallow; source and recorder share CoreConfig.
# Keep this process-local: never put locks or callbacks into serialized args.
_FEEDBACK_SOURCES = weakref.WeakValueDictionary()


def bind_feedback(core, source):
    if id(core) in _FEEDBACK_SOURCES:
        raise ValueError("A prompt feedback source is already bound to this configuration")
    _FEEDBACK_SOURCES[id(core)] = source


def unbind_feedback(core, source):
    if _FEEDBACK_SOURCES.get(id(core)) is source:
        del _FEEDBACK_SOURCES[id(core)]


class GradingRecorder(inference_records.Recorder):
    """Keep the reviewed grading policy in saved records and evidence identity."""

    def __init__(self, args, **kwargs):
        core = args.olmo_core
        self._grading_policy = (
            {
                "schema_version": grading.SCHEMA_VERSION,
                "reward_final_answer_only": True,
                "reward_reasoning_mode": getattr(core, "reward_reasoning_mode", "required"),
                "reward_zero_truncated": core.reward_zero_truncated,
            }
            if getattr(core, "reward_final_answer_only", False)
            else None
        )
        self._grading_inputs = []
        super().__init__(args, **kwargs)

    def record_group(self, samples, *, decision, reason=None):
        # The pinned recorder constructs and enqueues the row synchronously.
        # Capture per-response metadata before its whitelist drops these fields.
        self._grading_inputs = [(sample.metadata or {}).get("grading") for sample in samples]
        try:
            super().record_group(samples, decision=decision, reason=reason)
        finally:
            self._grading_inputs = []

    def _annotate_grading(self, row):
        if row.get("kind") == "group" and self._grading_policy is not None and "grading_policy" not in row:
            row["grading_policy"] = dict(self._grading_policy)
            # The pinned base protocol hashes the registry but not global grading
            # flags. Keep new-contract evidence separate without changing the
            # legacy recorder's manifest or default-off pooling identity.
            row["input_key"] = inference_records.sha256(
                {"input_key": row["input_key"], "grading_policy": self._grading_policy}
            )
            for response, selection in zip(row["responses"], self._grading_inputs, strict=True):
                if selection is not None:
                    response["grading"] = copy.deepcopy(selection)

    def _enqueue(self, row):
        self._annotate_grading(row)
        super()._enqueue(row)


class FeedbackRecorder(GradingRecorder):
    """Deliver scored groups before enqueueing; only feedback-dependent providers fail closed."""

    def __init__(self, args, source):
        self._source = weakref.ref(source)
        self._requires_feedback = source.provider.requires_feedback
        self._reward_key = getattr(args, "reward_key", None)
        self._zero_truncated = args.olmo_core.reward_zero_truncated
        self._probe = getattr(args.olmo_core, "prompt_replay_enabled", False) and getattr(
            args.olmo_core, "prompt_replay_answer_probe", False
        )
        self._answer_probes = []
        super().__init__(args, lineage_factory=lineage_identity)
        if self._requires_feedback and self.unavailable:
            raise RuntimeError("Prompt feedback requires an available inference recorder")

    def record_group(self, samples, *, decision, reason=None):
        if self._requires_feedback and self.unavailable:
            raise RuntimeError("Prompt feedback recorder became unavailable")
        self._answer_probes = []
        if self._probe:
            for sample in samples:
                targets = [
                    v.get("target") for v in (sample.metadata or {}).get("verifiers", []) if v.get("name") == "math"
                ]
                self._answer_probes.append(
                    replay_evidence.literal_answer_probe(sample.response, targets[0]) if len(targets) == 1 else None
                )
        # The pinned Recorder calls _enqueue synchronously once per group;
        # probes belong to that call and must not survive into the next group.
        failures = self.counts["failed"]
        super().record_group(samples, decision=decision, reason=reason)
        if self._requires_feedback and self.counts["failed"] != failures:
            raise RuntimeError("Could not construct prompt feedback record")

    def record_disposition(self, samples, *, accepted, staleness):
        # The discard callback may report this again; feedback is intentionally
        # idempotent so recorder and buffer terminal paths can both be observed.
        source = self._source()
        if source is not None and hasattr(source.provider, "commit") and not accepted:
            source.feedback(
                [{"kind": "disposition", "disposition": "discarded", "group_index": samples[0].group_index}]
            )
        super().record_disposition(samples, accepted=accepted, staleness=staleness)

    def _enqueue(self, row):
        self._annotate_grading(row)
        if row.get("kind") == "group":
            try:
                source = self._source()
                if source is None:
                    raise RuntimeError("Prompt feedback source is no longer available")
                row = row | {"reward_key": self._reward_key, "reward_zero_truncated": self._zero_truncated}
                if self._probe:
                    for response, probe in zip(row["responses"], self._answer_probes, strict=True):
                        response["literal_reference_in_text"] = probe
                    row["answer_probe_method"] = "literal_math_reference_v1"
                source.feedback([row])
            except Exception:
                if self._requires_feedback:
                    raise
                # Fixed mixtures only annotate the audit row. Preserve ordinary
                # best-effort recording without silently losing required feedback.
                self._count("failed")
                logger.exception("Inference record attribution unavailable for group %s", row.get("group_index"))
                return
        super()._enqueue(row)

    def metrics(self):
        source = self._source()
        return super().metrics() | (source.provider_metrics() if source is not None else {})


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
        "inventory_sha256": inference_records.sha256(identity),
        "basis": basis,
        "path": identity["path"],
        "served_path": str(path),
        "weights_hashed": False,
    }


def create_recorder(args):
    source = _FEEDBACK_SOURCES.get(id(args.olmo_core))
    if source is not None:
        return FeedbackRecorder(args, source)
    if getattr(args.olmo_core, "prompt_provider", "shuffle") == "remind" or getattr(
        args.olmo_core, "prompt_replay_enabled", False
    ):
        raise RuntimeError("Feedback recorder has no bound prompt feedback source")
    return GradingRecorder(args, lineage_factory=lineage_identity)
