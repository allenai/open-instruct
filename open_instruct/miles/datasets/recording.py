"""Connect MILES inference records to the model source prepared by this workflow.
The recorder receives a stable identity derived from the starting checkpoint's
file inventory, so separate runs prepared from the same source can share evidence
for prompt selection. The identity records its basis explicitly: it describes an
inventory rather than a hash of every weight tensor.
"""

import json
from pathlib import Path

from miles.utils import inference_records

from open_instruct.miles.execution import workflow

MARKER = "workflow-model.json"


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
    return inference_records.Recorder(args, lineage_factory=lineage_identity)
