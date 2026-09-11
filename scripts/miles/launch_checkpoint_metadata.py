"""Read-only inspection of an existing checkpoint's metadata on Saturn."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

SOURCE = "/weka/oe-training-default/robertb/open-instruct/control-exercise/01M28PAQW8YCJ6P8M9F2343PYH/controls/metrics/core/rollout_0000003/model/.metadata"
PROBE = r"""
import collections
import hashlib
import json
import math
import pickle
import re
import shutil
import time
from pathlib import Path

import torch
from torch.distributed.tensor import Shard
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset
from olmo_core.distributed.checkpoint.contiguous_planner import contiguous_shard_metadata

source = Path(__SOURCE__)
raw = source.read_bytes()
metadata = pickle.loads(raw)
by_rank = collections.defaultdict(lambda: {"written_bytes": 0, "items": 0, "files": set()})
for index, item in metadata.storage_data.items():
    rank = re.match(r"__(\d+)_", item.relative_path).group(1)
    by_rank[rank]["written_bytes"] += item.length
    by_rank[rank]["items"] += 1
    by_rank[rank]["files"].add(item.relative_path)
for value in by_rank.values():
    value["files"] = len(value["files"])
shapes = collections.Counter()
for value in metadata.state_dict_metadata.values():
    if hasattr(value, "size"):
        shapes[tuple(value.size)] += 1
report = {"source": str(source), "metadata_sha256": hashlib.sha256(raw).hexdigest(),
          "by_rank": dict(by_rank), "tensor_shape_counts": {str(k):v for k,v in shapes.items()},
          "planning_probe": [],
          "runtime_lock": json.loads(Path("/opt/core-rl/build/runtime/miles/runtime.lock.json").read_text())}
Path("/output").mkdir(exist_ok=True)
shutil.copyfile(source, "/output/native.metadata")
torch.set_num_threads(2)
for shape in sorted(shapes, key=math.prod, reverse=True)[:3]:
    if len(shape) != 1 or shape[0] > 2_000_000_000:
        continue
    # Probe ordinary flat 2-way sharding; the original mesh is not stored in DCP metadata.
    for mesh, placements in (((2,), (Shard(0),)), ((2, 1), (Shard(0), Shard(0)))):
        coordinate = [0] * len(mesh)
        start = time.perf_counter()
        expected = _compute_local_shape_and_global_offset(shape, mesh, coordinate, placements)
        legacy_seconds = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(1000):
            actual = contiguous_shard_metadata(shape, mesh, coordinate, placements)
        candidate_seconds = (time.perf_counter() - start) / 1000
        assert actual == expected
        report["planning_probe"].append({"shape":shape, "mesh":mesh, "legacy_seconds":legacy_seconds,
                                         "candidate_seconds":candidate_seconds, "metadata_exact":True})
Path("/output/metadata-profile.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--source", default=SOURCE)
    args = parser.parse_args()
    document = {
        "version": "v2",
        "description": "Read-only checkpoint shard ownership and metadata-planning probe",
        "tasks": [
            {
                "name": "checkpoint-metadata",
                "image": {"beaker": args.image},
                "command": ["python", "-c"],
                "arguments": [PROBE.replace("__SOURCE__", repr(args.source))],
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "resources": {"cpuCount": 4, "memory": "32 GiB"},
                "constraints": {"cluster": ["ai2/saturn"]},
                "context": {"priority": "urgent", "minRuntime": "5m", "autoResume": False},
                "timeout": "20m",
            }
        ],
    }
    with tempfile.TemporaryDirectory(prefix="checkpoint-metadata-") as temporary:
        path = Path(temporary) / "experiment.json"
        path.write_text(json.dumps(document))
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
