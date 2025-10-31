#!/usr/bin/env python3
"""
Fetch prediction artifacts for specific Beaker experiments.

This mirrors the data loading workflow in Adapt Leaderboard's ComparisonModal:
  * look up each experiment via ``beaker experiment get``
  * download the backing dataset with ``beaker dataset fetch``
  * read the ``*-requests.jsonl`` and ``*-predictions.jsonl`` files
  * deduplicate entries by ``native_id`` so paired comparisons are stable

No existing scripts are modified. The goal is to provide a drop-in utility
that skips search heuristics and lets you target exact experiment IDs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


JsonDict = MutableMapping[str, Any]
JsonList = List[JsonDict]


class BeakerCommandError(RuntimeError):
    """Raised when a Beaker CLI command fails."""

    def __init__(self, command: Sequence[str], stdout: str, stderr: str, exit_code: int):
        super().__init__(f"Command {' '.join(command)} failed with exit code {exit_code}")
        self.command = list(command)
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


def run_beaker_command(args: Sequence[str]) -> str:
    """Execute a beaker CLI command and return stdout or raise on failure."""
    process = subprocess.run(
        args,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise BeakerCommandError(args, process.stdout, process.stderr, process.returncode)
    return process.stdout


def sanitize_name(value: str) -> str:
    """Return a filesystem-friendly name derived from value."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return sanitized or "experiment"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def valid_jsonl_files(path: Path) -> bool:
    return any(file.suffix == ".jsonl" for file in path.glob("*.jsonl"))


@dataclass
class ExperimentDataset:
    experiment_id: str
    experiment_name: str
    dataset_id: str
    download_dir: Path
    requests: JsonList
    predictions: JsonList
    paired: List[Dict[str, Any]]


def get_experiment_details(experiment_id: str) -> JsonDict:
    """Return the raw experiment details from the Beaker CLI."""
    stdout = run_beaker_command(["beaker", "experiment", "get", experiment_id, "--format", "json"])
    experiments = json.loads(stdout)
    if not experiments:
        raise RuntimeError(f"Experiment {experiment_id} not found.")
    experiment = experiments[0]
    if not isinstance(experiment, dict):
        raise RuntimeError(f"Unexpected payload for experiment {experiment_id}: {type(experiment)!r}")
    return experiment


def dataset_id_from_experiment(experiment: Mapping[str, Any]) -> str:
    try:
        jobs = experiment["jobs"]
        if not jobs:
            raise KeyError
        return jobs[0]["execution"]["result"]["beaker"]
    except (KeyError, TypeError, IndexError):
        raise RuntimeError(f"Unable to locate dataset ID in experiment payload {experiment.get('id')}")


def download_dataset(dataset_id: str, cache_root: Path, overwrite: bool = False) -> Path:
    cache_dir = ensure_directory(cache_root / dataset_id)
    has_jsonl = valid_jsonl_files(cache_dir)
    if has_jsonl and not overwrite:
        return cache_dir

    if cache_dir.exists() and overwrite:
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset {dataset_id} into {cache_dir}")
    run_beaker_command(["beaker", "dataset", "fetch", dataset_id, "--output", str(cache_dir)])
    return cache_dir


def sanitize_json_line(line: str) -> str:
    """Replace NaN/Infinity placeholders with JSON-safe values."""
    replacements = {
        ": NaN": ": null",
        ":NaN": ":null",
        ": Infinity": ": null",
        ":Infinity": ":null",
        ":-Infinity": ":null",
        ": -Infinity": ": null",
    }
    for needle, replacement in replacements.items():
        if needle in line:
            line = line.replace(needle, replacement)
    return line


def read_jsonl_file(path: Path) -> JsonList:
    records: JsonList = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            sanitized = sanitize_json_line(line)
            try:
                payload = json.loads(sanitized)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse JSONL line in {path}: {exc}\nLine: {line[:200]}") from exc
            records.append(payload)
    return records


def read_dataset_entries(dataset_dir: Path) -> Dict[str, JsonList]:
    entries: Dict[str, JsonList] = {}
    for jsonl_path in sorted(dataset_dir.glob("*.jsonl")):
        entries[jsonl_path.name] = read_jsonl_file(jsonl_path)
    return entries


def merge_duplicates(items: Iterable[JsonDict]) -> JsonList:
    merged: JsonList = []
    seen_native_ids: set[Any] = set()
    for item in items:
        native_id = item.get("native_id")
        if native_id is not None and native_id in seen_native_ids:
            continue
        merged.append(item)
        if native_id is not None:
            seen_native_ids.add(native_id)
    return merged


def build_pairs(requests: Iterable[JsonDict], predictions: Iterable[JsonDict]) -> List[Dict[str, Any]]:
    prediction_map = {
        pred.get("native_id"): pred
        for pred in predictions
        if pred.get("native_id") is not None
    }
    pairs: List[Dict[str, Any]] = []
    for request in requests:
        native_id = request.get("native_id")
        if native_id is None:
            continue
        prediction = prediction_map.get(native_id)
        if prediction is None:
            continue
        pairs.append(
            {
                "native_id": native_id,
                "request": request,
                "prediction": prediction,
            }
        )
    return pairs


def collect_experiment_dataset(
    experiment_id: str,
    cache_root: Path,
    overwrite: bool = False,
) -> ExperimentDataset:
    experiment = get_experiment_details(experiment_id)
    dataset_id = dataset_id_from_experiment(experiment)
    experiment_name = experiment.get("name", f"experiment-{experiment_id}")
    dataset_dir = download_dataset(dataset_id, cache_root=cache_root, overwrite=overwrite)
    entries = read_dataset_entries(dataset_dir)

    request_entries: JsonList = []
    prediction_entries: JsonList = []
    for filename, records in entries.items():
        if filename.endswith("-requests.jsonl"):
            request_entries.extend(records)
        elif filename.endswith("-predictions.jsonl"):
            prediction_entries.extend(records)

    request_entries = merge_duplicates(request_entries)
    prediction_entries = merge_duplicates(prediction_entries)
    paired = build_pairs(request_entries, prediction_entries)

    return ExperimentDataset(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        dataset_id=dataset_id,
        download_dir=dataset_dir,
        requests=request_entries,
        predictions=prediction_entries,
        paired=paired,
    )


def write_experiment_payload(experiment: ExperimentDataset, output_dir: Path) -> Path:
    safe_name = sanitize_name(experiment.experiment_name)
    target_dir = ensure_directory(output_dir / f"{safe_name}-{experiment.experiment_id}")
    payload = {
        "experiment_id": experiment.experiment_id,
        "experiment_name": experiment.experiment_name,
        "dataset_id": experiment.dataset_id,
        "source_directory": str(experiment.download_dir),
        "request_count": len(experiment.requests),
        "prediction_count": len(experiment.predictions),
        "paired_count": len(experiment.paired),
        "requests": experiment.requests,
        "predictions": experiment.predictions,
        "paired": experiment.paired,
    }
    output_path = target_dir / "predictions.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch predictions for exact Beaker experiment IDs.",
    )
    parser.add_argument(
        "--experiment-id",
        dest="experiment_ids",
        action="append",
        required=True,
        help="Experiment ID to fetch. Provide multiple times for multiple experiments.",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache/datasets",
        type=Path,
        help="Directory used to cache downloaded datasets (default: cache/datasets).",
    )
    parser.add_argument(
        "--output-dir",
        default="fetched_predictions",
        type=Path,
        help="Directory where parsed prediction JSON will be written (default: fetched_predictions).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download datasets even if cached copies exist.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cache_root = ensure_directory(Path(args.cache_dir).expanduser().resolve())
    output_root = ensure_directory(Path(args.output_dir).expanduser().resolve())

    all_payloads: List[Path] = []
    for experiment_id in args.experiment_ids:
        print(f"\n=== Fetching experiment {experiment_id} ===")
        try:
            experiment_dataset = collect_experiment_dataset(
                experiment_id=experiment_id,
                cache_root=cache_root,
                overwrite=args.overwrite,
            )
        except BeakerCommandError as err:
            sys.stderr.write(f"Beaker command failed ({' '.join(err.command)}):\n{err.stderr}\n")
            return err.exit_code
        except Exception as exc:  # noqa: BLE001 - surface unexpected errors
            sys.stderr.write(f"Error while processing {experiment_id}: {exc}\n")
            return 1

        output_path = write_experiment_payload(experiment_dataset, output_dir=output_root)
        all_payloads.append(output_path)
        print(f"  Cached dataset in {experiment_dataset.download_dir}")
        print(f"  Wrote parsed predictions to {output_path}")

    if len(all_payloads) > 1:
        summary_path = output_root / "summary.json"
        summary_payload = {
            "experiments": [
                json.loads(path.read_text(encoding="utf-8"))
                for path in all_payloads
            ],
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2)
        print(f"\nWrote multi-experiment summary to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
