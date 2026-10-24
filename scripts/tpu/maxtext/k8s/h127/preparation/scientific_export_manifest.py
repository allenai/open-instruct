"""Snapshot finalized H127 scientific checkpoint objects; never modify GCS."""

import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

GCLOUD = "/snap/google-cloud-cli/current/bin/gcloud"
ROOT = "gs://ai2-olmo-dev/post-training/runs/robertb/presentation14t-v1/"


def build_manifest(prefix, step, objects, metadata):
    assert {"_CHECKPOINT_METADATA", "commit_success.txt"} <= objects.keys()
    assert not any("orbax-checkpoint-tmp" in name for name in objects)
    assert metadata["commit_timestamp_nsecs"] > metadata["init_timestamp_nsecs"] > 0
    assert {"model_params", "optimizer_state"} <= set(metadata["item_handlers"])
    assert any(name.startswith("optimizer_state/") for name in objects)
    params = {
        name.removeprefix("model_params/"): entry
        for name, entry in objects.items()
        if name.startswith("model_params/")
    }
    assert len(params) == 197, "Inspect unexpected model checkpoint layout before export"
    return {
        "source_prefix": prefix + "model_params/",
        "step": step,
        "checkpoint_finalized": True,
        "checkpoint_metadata": metadata,
        "checkpoint_metadata_object": objects["_CHECKPOINT_METADATA"],
        "commit_success_object": objects["commit_success.txt"],
        "objects": params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("control", "uniform"), required=True)
    parser.add_argument("--step", type=int, choices=(512, 2048), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Refusing to overwrite an existing manifest"
    prefix = ROOT + f"robertb-pres14t-v1-{args.arm}-screen/checkpoints/{args.step}/"

    def listing():
        rows = json.loads(subprocess.check_output([GCLOUD, "storage", "ls", "--json", prefix + "**"], text=True))
        objects = {}
        for row in rows:
            assert row["type"] == "cloud_object"
            metadata = row["metadata"]
            url = "gs://" + metadata["bucket"] + "/" + metadata["name"]
            assert url.startswith(prefix)
            name = url.removeprefix(prefix)
            assert name not in objects
            objects[name] = {
                "size": int(metadata["size"]),
                "crc32c": metadata["crc32c"],
                "generation": metadata["generation"],
            }
        return objects

    objects = listing()
    raw = subprocess.check_output([GCLOUD, "storage", "cat", prefix + "_CHECKPOINT_METADATA"])
    result = build_manifest(prefix, args.step, objects, json.loads(raw))
    assert listing() == objects, "Checkpoint changed during snapshot"
    result.update(
        metadata_sha256=hashlib.sha256(raw).hexdigest(),
        captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"source": result["source_prefix"], "model_objects": len(result["objects"]), "finalized": True}))


if __name__ == "__main__":
    main()
