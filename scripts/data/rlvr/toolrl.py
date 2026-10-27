"""Convert the pinned ToolRL 4K source to prompt-only Open Instruct RL JSONL.

Run from the repository root with ``python -m scripts.data.rlvr.toolrl --help``.
Only preparation is performed: no tools, training, or Hub uploads are invoked.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from urllib import request

from open_instruct import toolrl_utils

REVISION = "8cee13ec0ca72f0461da372a93a6fd8140dbb840"
SOURCE_PATH = "dataset/rlla_4k_raw/rlla_rl.json"
SOURCE_URL = f"https://raw.githubusercontent.com/qiancheng0/ToolRL/{REVISION}/{SOURCE_PATH}"
SOURCE_SHA256 = "f61cf5820afa8ced1a7fcc6b5078d3ccdfdf85623e7ec70783a1a63591620ff0"


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def convert_rows(rows, *, eval_count=80, seed=31415, think_prefix_in_prompt=False):
    """Retain structurally valid examples; identical inputs stay in one partition.

    Unlike upstream's row shuffle, grouping also prevents identical dialogue inputs
    with different tool menus from crossing the train/eval boundary. Whole groups
    can make the requested eval count slightly larger. This is a new split, not a
    reproduction of the paper's split or a decontamination check against BFCL.
    """
    if type(eval_count) is not int or not 0 < eval_count < len(rows):
        raise ValueError("eval_count must be positive and smaller than the source row count")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    groups = {}
    converted = []
    rejected = []
    call_counts = Counter()
    verifier = "toolrl_think_prefix" if think_prefix_in_prompt else "toolrl"
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip() for key in ("instruction", "input", "output")
        ):
            raise ValueError(f"Row {index}: expected nonempty instruction, input, and output strings")
        try:
            _, calls = toolrl_utils.reference_parts(row["output"])
        except ValueError as error:
            rejected.append(
                {"source_row": index, "reason": str(error), "reference_sha256": sha(row["output"].encode())}
            )
            continue
        group = sha(row["input"].encode())
        groups.setdefault(group, []).append(index)
        call_counts[len(calls)] += 1
        converted.append(
            {
                "messages": [
                    {"role": "system", "content": row["instruction"]},
                    {"role": "user", "content": row["input"]},
                ],
                "ground_truth": row["output"],
                "dataset": "toolrl",
                "metadata": {
                    "prepared_sample_id": f"toolrl:{REVISION}:{index}",
                    "source_revision": REVISION,
                    "source_row": index,
                    "dialogue_group": group,
                    "reference_call_count": len(calls),
                    "verifiers": [{"name": verifier, "target": row["output"], "weight": 1.0}],
                },
            }
        )
    if eval_count >= len(converted):
        raise ValueError("Too few valid references for the requested eval_count")
    held_out = set()
    for group in sorted(groups, key=lambda key: sha(f"{seed}:{key}".encode())):
        if len(held_out) >= eval_count:
            break
        held_out.update(groups[group])
    if len(held_out) == len(converted):
        raise ValueError("Grouping leaves no training rows; reduce eval_count or use more distinct dialogues")
    partitions = {"train": [], "eval": []}
    for row in converted:
        partitions["eval" if row["metadata"]["source_row"] in held_out else "train"].append(row)
    summary = {
        "source_rows": len(rows),
        "retained_rows": len(converted),
        "rejected_rows": rejected,
        "unique_dialogues": len(groups),
        "reference_call_counts": dict(sorted(call_counts.items())),
        "partition_rows": {split: len(values) for split, values in partitions.items()},
        "eval_count_requested": eval_count,
        "split_seed": seed,
        "split_method": "sha256-seeded-dialogue-groups-v1",
        "verifier": verifier,
        "bfcl_overlap_audited": False,
    }
    return partitions, summary


def prepare(source: bytes, output: Path, **options):
    if sha(source) != SOURCE_SHA256:
        raise ValueError("Source SHA256 does not match the pinned ToolRL release")
    rows = json.loads(source)
    if not isinstance(rows, list) or len(rows) != 4000:
        raise ValueError("Expected the pinned 4000-row ToolRL source")
    partitions, summary = convert_rows(rows, **options)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        artifacts = {}
        for split, values in partitions.items():
            raw = b"".join(encoded(row) for row in values)
            filename = f"{split}.jsonl"
            (staging / filename).write_bytes(raw)
            artifacts[split] = {"path": filename, "sha256": sha(raw), "records": len(values)}
        manifest = {
            "schema_version": 1,
            "source": {"url": SOURCE_URL, "revision": REVISION, "sha256": SOURCE_SHA256},
            "summary": summary,
            "artifacts": artifacts,
            "miles": {
                "input_key": "messages",
                "metadata_key": "metadata",
                "label_key": "ground_truth",
                "apply_chat_template": True,
                # Required compatibility identifier for the existing manifest importer;
                # the importer resolves current open_instruct verifier factories.
                "custom_rm_path": "olmo_miles.rl.rewards.registered_reward",
            },
        }
        (staging / "rl-manifest.json").write_bytes(encoded(manifest))
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Local copy of pinned rlla_rl.json; otherwise download it")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument(
        "--think-prefix-in-prompt",
        action="store_true",
        help="Select the verifier for a model chat template whose generation prompt ends in <think>",
    )
    args = parser.parse_args()
    if args.input:
        source = args.input.read_bytes()
    else:
        with request.urlopen(SOURCE_URL, timeout=60) as response:
            source = response.read()
    manifest = prepare(
        source,
        args.output_dir,
        eval_count=args.eval_count,
        seed=args.seed,
        think_prefix_in_prompt=args.think_prefix_in_prompt,
    )
    print(json.dumps(manifest["summary"], indent=2))
    print(f"MILES data.rl_manifest: {args.output_dir.resolve() / 'rl-manifest.json'}")


if __name__ == "__main__":
    main()
