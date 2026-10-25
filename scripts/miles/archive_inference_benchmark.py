"""Download every finalized attempt, full Beaker results/logs, and a hash inventory."""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def beaker_json(*arguments):
    return json.loads(subprocess.check_output(["beaker", *arguments, "--format", "json"], text=True))


def archive(experiment, destination):
    destination.mkdir(parents=True, exist_ok=True)
    metadata = beaker_json("experiment", "get", experiment)
    (destination / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    record = metadata[0] if isinstance(metadata, list) else metadata
    attempts = []
    for job in record["jobs"]:
        job_id = job["id"]
        details = beaker_json("job", "get", job_id)
        current = details[0] if isinstance(details, list) else details
        directory = destination / "attempts" / job_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "job.json").write_text(json.dumps(details, indent=2) + "\n")
        if not current["status"].get("finalized"):
            attempts.append({"job": job_id, "downloaded": False, "reason": "not_finalized"})
            continue
        with (directory / "full.log").open("w") as stream:
            subprocess.run(["beaker", "job", "logs", job_id], stdout=stream, check=True)
        dataset = current.get("result", {}).get("beaker")
        if not dataset:
            raise ValueError(f"Finalized attempt {job_id} has no result dataset")
        subprocess.run(["beaker", "dataset", "fetch", dataset, "--output", str(directory / "result")], check=True)
        attempts.append({"job": job_id, "dataset": dataset, "downloaded": True})
    files = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path == destination / "archive-manifest.json":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {"path": str(path.relative_to(destination)), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        )
    manifest = {
        "experiment": record["id"],
        "archived_utc": datetime.now(UTC).isoformat(),
        "attempts": attempts,
        "files": files,
        "scope": "All files in each finalized Beaker result plus full job logs; WEKA files omitted by the run collector require separate recovery.",
    }
    (destination / "archive-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"experiment": record["id"], "attempts": attempts, "files": len(files)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    archive(args.experiment, args.destination)


if __name__ == "__main__":
    main()
