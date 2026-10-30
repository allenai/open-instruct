#!/usr/bin/env python3
"""Audit IF training JSONL; add --output to write only admitted rows.

Supports MILES metadata.verifiers and legacy dataset/ground_truth routing. Custom
verifier aliases must be explicitly trusted with --verifier-name. Rows are never
rewritten, and existing files (including path aliases) are never overwritten.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

# Direct execution must use this checkout, even when an editable installation
# points at a different worktree.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from open_instruct import ifeval_constraints  # noqa: E402


def _targets(row):
    """Validate the routing envelope and return (verifier name, target) pairs."""
    if not isinstance(row, dict):
        raise ValueError("row must be a JSON object")
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and "verifiers" in metadata:
        specs = metadata["verifiers"]
        if not isinstance(specs, list) or not specs:
            raise ValueError("metadata.verifiers must be a nonempty array")
        result = []
        for index, spec in enumerate(specs):
            if not isinstance(spec, dict) or not isinstance(spec.get("name"), str) or not spec["name"].strip():
                raise ValueError(f"metadata.verifiers[{index}] requires a nonempty name")
            if "target" not in spec:
                raise ValueError(f"metadata.verifiers[{index}] is missing target")
            result.append((spec["name"], spec["target"]))
        return result
    if "dataset" not in row or "ground_truth" not in row:
        raise ValueError("unsupported row envelope: require metadata.verifiers or dataset and ground_truth")
    labels, targets = row["dataset"], row["ground_truth"]
    if isinstance(labels, str) and labels.strip():
        if isinstance(targets, list):
            if len(targets) != 1:
                raise ValueError("scalar dataset requires exactly one ground_truth target")
            targets = targets[0]
        return [(labels, targets)]
    if not isinstance(labels, list) or not labels:
        raise ValueError("dataset must be a nonempty verifier name or array of verifier names")
    if any(not isinstance(name, str) or not name.strip() for name in labels):
        raise ValueError("dataset entries must be nonempty verifier names")
    if not isinstance(targets, list) or len(labels) != len(targets):
        raise ValueError("dataset and ground_truth arrays must have matching lengths")
    return list(zip(labels, targets))


def _check_paths(input_path, report_path, output_path):
    paths = [input_path, report_path] + ([output_path] if output_path is not None else [])
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("input, report and output paths must be distinct (including symlink aliases)")
    for path in paths[1:]:
        if path.exists() or path.is_symlink():
            raise ValueError(f"refusing to overwrite existing file: {path}")
        if not path.parent.is_dir():
            raise ValueError(f"destination directory does not exist: {path.parent}")


def _publish(temp_path, destination):
    # Atomic installation fails if another process created the destination after
    # validation. Each temporary file is on its destination's filesystem.
    os.link(temp_path, destination)


def audit(input_path, report_path, *, output_path=None, verifier_names=("ifeval",)):
    """Stream an audit to JSON, optionally writing a filtered JSONL file."""
    input_path, report_path = Path(input_path), Path(report_path)
    output_path = Path(output_path) if output_path is not None else None
    _check_paths(input_path, report_path, output_path)
    names = set(verifier_names)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("verifier_names must contain nonempty trusted verifier names")
    legacy_names = {name.lower() for name in names}
    counts = {"total_rows": 0, "if_rows": 0, "kept_rows": 0, "dropped_rows": 0}
    reasons = Counter()
    input_hash, kept_hash = hashlib.sha256(), hashlib.sha256()
    temp_paths, published = [], []
    try:
        with ExitStack() as stack:
            dropped = stack.enter_context(tempfile.TemporaryFile(mode="w+t", encoding="utf-8"))
            output = None
            if output_path is not None:
                output = stack.enter_context(
                    tempfile.NamedTemporaryFile(mode="wb", dir=output_path.parent, delete=False)
                )
                temp_paths.append(Path(output.name))
            with input_path.open("rb") as source:
                for line_number, raw in enumerate(source, 1):
                    input_hash.update(raw)
                    if not raw.strip():
                        kept_hash.update(raw)
                        if output is not None:
                            output.write(raw)
                        continue
                    row_index = counts["total_rows"]
                    counts["total_rows"] += 1
                    try:
                        row = json.loads(raw.decode("utf-8"))
                        targets = _targets(row)
                    except (ValueError, UnicodeError) as error:
                        raise ValueError(f"{input_path}: line {line_number} (row {row_index}): {error}") from error
                    row_reasons = []
                    is_if = False
                    metadata = row.get("metadata")
                    legacy = not (isinstance(metadata, dict) and "verifiers" in metadata)
                    for name, target in targets:
                        # Legacy reward routing lowercases names; MILES registry
                        # keys are exact, including explicitly trusted aliases.
                        matches = name.lower() in legacy_names if legacy else name in names
                        if matches:
                            is_if = True
                            row_reasons.extend(ifeval_constraints.if_training_issues(target))
                    counts["if_rows"] += int(is_if)
                    row_reasons = list(dict.fromkeys(row_reasons))
                    if row_reasons:
                        counts["dropped_rows"] += 1
                        reasons.update(row_reasons)
                        record = {"row_index": row_index, "line_number": line_number, "reasons": row_reasons}
                        if "id" in row:
                            record["id"] = row["id"]
                        elif "sample_id" in row:
                            record["id"] = row["sample_id"]
                        elif isinstance(row.get("metadata"), dict) and "sample_id" in row["metadata"]:
                            record["id"] = row["metadata"]["sample_id"]
                        metadata = row.get("metadata")
                        for key in ("prepared_sample_id", "custom_id"):
                            if key in row:
                                record[key] = row[key]
                            elif isinstance(metadata, dict) and key in metadata:
                                record[key] = metadata[key]
                        dropped.write(json.dumps(record, ensure_ascii=False) + "\n")
                    else:
                        counts["kept_rows"] += 1
                        kept_hash.update(raw)
                        if output is not None:
                            output.write(raw)
            if output is not None:
                output.flush()
            summary = {
                "schema_version": 1,
                "input": str(input_path.resolve()),
                "output": str(output_path.resolve()) if output_path is not None else None,
                "input_sha256": input_hash.hexdigest(),
                "output_sha256": kept_hash.hexdigest() if output_path is not None else None,
                "kept_sha256": kept_hash.hexdigest(),
                "verifier_names": sorted(names),
                **counts,
                "reason_counts": dict(sorted(reasons.items())),
            }
            report = stack.enter_context(
                tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=report_path.parent, delete=False)
            )
            temp_paths.append(Path(report.name))
            # Spool dropped rows so memory use does not grow with input size.
            report.write(json.dumps(summary, ensure_ascii=False, indent=2)[:-2] + ',\n  "dropped": [')
            dropped.seek(0)
            for index, record in enumerate(dropped):
                report.write((",\n" if index else "\n") + "    " + record.rstrip("\n"))
            report.write("\n  ]\n}\n")
        if output_path is not None:
            _publish(temp_paths[0], output_path)
            published.append(output_path)
        _publish(temp_paths[-1], report_path)
        published.append(report_path)
        return summary
    except BaseException:
        for path in published:
            path.unlink()
        raise
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL (read only)")
    parser.add_argument("--report", required=True, type=Path, help="New JSON audit report")
    parser.add_argument("--output", type=Path, help="New filtered JSONL; omitted means audit only")
    parser.add_argument(
        "--verifier-name", action="append", help="Trust this verifier name as IFEval; repeatable (default: ifeval)"
    )
    args = parser.parse_args(argv)
    try:
        summary = audit(
            args.input, args.report, output_path=args.output, verifier_names=args.verifier_name or ["ifeval"]
        )
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
