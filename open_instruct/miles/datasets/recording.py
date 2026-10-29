"""Connect MILES inference records to the model source prepared by this workflow.
The recorder receives a stable identity derived from the starting checkpoint's
file inventory, so separate runs prepared from the same source can share evidence
for prompt selection. The identity records its basis explicitly: it describes an
inventory rather than a hash of every weight tensor.
"""

import json
import weakref
from pathlib import Path

from miles.utils import inference_records

from open_instruct.miles.execution import workflow

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


class FeedbackRecorder(inference_records.Recorder):
    """Deliver scored groups before the best-effort disk queue, including drops."""

    def __init__(self, args, source):
        self._source = weakref.ref(source)
        self._reward_key = getattr(args, "reward_key", None)
        self._zero_truncated = args.olmo_core.reward_zero_truncated
        super().__init__(args, lineage_factory=lineage_identity)
        if self.unavailable:
            raise RuntimeError("Prompt feedback requires an available inference recorder")

    def record_group(self, samples, *, decision, reason=None):
        if self.unavailable:
            raise RuntimeError("Prompt feedback recorder became unavailable")
        failures = self.counts["failed"]
        super().record_group(samples, decision=decision, reason=reason)
        if self.counts["failed"] != failures:
            raise RuntimeError("Could not construct prompt feedback record")

    def _enqueue(self, row):
        if row.get("kind") == "group":
            source = self._source()
            if source is None:
                raise RuntimeError("Prompt feedback source is no longer available")
            row = row | {"reward_key": self._reward_key, "reward_zero_truncated": self._zero_truncated}
            source.feedback([row])
        super()._enqueue(row)


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
    if getattr(args.olmo_core, "prompt_provider", "shuffle") == "remind":
        raise RuntimeError("ReMind recorder has no bound prompt feedback source")
    return inference_records.Recorder(args, lineage_factory=lineage_identity)
