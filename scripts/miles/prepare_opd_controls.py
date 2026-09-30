"""Freeze captured evaluation prompts, completed checkpoints and two exposure schedules."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.miles import opd_control


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(config, output):
    output.mkdir(parents=True, exist_ok=False)
    capture = Path(config["capture"])
    rows = []
    for line in capture.read_text().splitlines():
        item = json.loads(line)
        length = item["response_length"]
        if not 0 < length < len(item["tokens"]):
            raise ValueError("Ambiguous captured prompt boundary")
        rows.append(
            dict(
                dataset=item["dataset"],
                question_id=f"{item['dataset']}:{item['sample_index']}",
                prompt=item["prompt"],
                label=item["label"],
                input_ids=item["tokens"][:-length],
            )
        )
    counts = Counter(row["dataset"] for row in rows)
    if counts != {"dapo_math_holdout": 512, "math_aime_2025": 30, "math_brumo_2025": 30}:
        raise ValueError(f"Unexpected panel: {counts}")
    if len({row["question_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate question identity")
    panel = output / "panel.jsonl"
    panel.write_text("".join(json.dumps(row) + "\n" for row in sorted(rows, key=lambda row: row["question_id"])))
    (output / "verifiers.json").write_bytes(Path(config["registry"]).read_bytes())
    manifest = dict(
        config=config,
        counts=dict(counts),
        capture_sha256=digest(capture),
        panel_sha256=digest(panel),
        verifiers_sha256=digest(output / "verifiers.json"),
        checkpoints={},
    )
    for name, location in config["checkpoints"].items():
        root = Path(location)
        if not (root / ".complete").is_file():
            raise ValueError(f"Incomplete checkpoint: {root}")
        files = [root / "config.json", *sorted(root.glob("*.safetensors"))]
        if len(files) < 2:
            raise ValueError("Missing checkpoint weights")
        manifest["checkpoints"][name] = dict(
            path=str(root), files={p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in files}
        )
    trace = Path(config["trace"])
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    schedules = opd_control.schedules(events)
    source = Path(config["training_data"])
    source_rows = [json.loads(line) for line in source.read_text().splitlines()]
    cohorts = opd_control.resolve_cohorts(source_rows, events)
    manifest["exposure"] = dict(trace_sha256=digest(trace), training_data_sha256=digest(source), cohorts={})
    for name, keys in schedules.items():
        path = output / f"{name}.jsonl"
        cohort = cohorts[name]
        path.write_text("".join(json.dumps(row) + "\n" for row in cohort["rows"]))
        manifest["exposure"]["cohorts"][name] = dict(
            sha256=digest(path),
            groups=len(keys),
            prompt_hashes=keys,
            source_row_indices=cohort["source_row_indices"],
            original_group_indices=cohort["group_indices"],
        )
    manifest["exposure"]["shared_prompts"] = len(set(schedules["admitted"]) & set(schedules["selected"]))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "counts": counts,
                "checkpoints": list(manifest["checkpoints"]),
                "shared_training_prompts": manifest["exposure"]["shared_prompts"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()), args.output)
