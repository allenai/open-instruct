"""Materialize a deterministic mixture without changing source row identities."""

import copy
import hashlib
import json
from pathlib import Path

SOURCES = {"gsm8k": "gsm8k", "math": "math", "ifeval": "ifeval_old"}
UPDATES = 2
PROMPTS_PER_SOURCE = 2
SAMPLES_PER_PROMPT = 4
BATCH_PROMPTS = len(SOURCES) * PROMPTS_PER_SOURCE


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_source(root, source, template_hash):
    manifest_bytes = (root / "preparation.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    template = manifest.get("template_sha256") or manifest.get("descriptor", {}).get("template_sha256")
    if template != template_hash:
        raise ValueError(f"{source}: source and mixture chat templates differ")
    provenance = manifest.get("source") or manifest
    if isinstance(provenance, dict) and provenance.get("kind") == "local_jsonl":
        if len(provenance.get("sha256", "")) != 64:
            raise ValueError(f"{source}: offline snapshot requires a content hash")
    elif not isinstance(provenance, dict) or len(provenance.get("revision", "")) != 40:
        raise ValueError(f"{source}: source requires a pinned dataset revision or honest offline snapshot")
    partitions = {}
    hashes = {}
    for split in ("train", "eval"):
        raw = (root / f"{split}.jsonl").read_bytes()
        expected = manifest.get("prepared_sha256", {}).get(split) or manifest.get("files", {}).get(f"{split}.jsonl")
        if expected is None or digest(raw) != expected:
            raise ValueError(f"{source}: missing or changed prepared {split} hash")
        hashes[split] = expected
        partitions[split] = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        for row in partitions[split]:
            metadata = row["metadata"]
            verifiers = metadata.get("verifiers", [])
            if (
                not isinstance(row["input"], str)
                or not row["input"]
                or not isinstance(row["label"], str)
                or len(verifiers) != 1
                or verifiers[0].get("name") != SOURCES[source]
                or verifiers[0].get("target") != row["label"]
                or verifiers[0].get("weight", 1.0) != 1.0
                or "mixture" in metadata
            ):
                raise ValueError(f"{source}: invalid source prompt, verifier, or target")
    return partitions, {
        "root": str(root.resolve()),
        "preparation_sha256": digest(manifest_bytes),
        "prepared_sha256": hashes,
        "preparation": manifest,
    }


def select_rows(rows, *, source, split, count, seed, tokenizer, max_prompt_tokens):
    ranked = []
    seen = set()
    for position, original in enumerate(rows):
        original_id = original["metadata"].get("prepared_sample_id") or digest(encoded(original))
        if not isinstance(original_id, str) or original_id in seen:
            raise ValueError(f"{source}/{split}: duplicate or invalid source identity")
        seen.add(original_id)
        ids = tokenizer.encode(original["input"], add_special_tokens=False)
        if not 0 < len(ids) <= max_prompt_tokens:
            continue
        row = copy.deepcopy(original)
        key = f"{source}:{original_id}"
        row["metadata"]["mixture"] = {
            "source": source,
            "id": key,
            "source_identity": original_id,
            "source_split": split,
            "source_position": position,
            "source_row_sha256": digest(encoded(original)),
            "prompt_tokens": len(ids),
            "token_ids_sha256": digest(encoded(ids)),
        }
        priority = digest(encoded([seed, source, split, original_id]))
        ranked.append((priority, key, row))
    if len(ranked) < count:
        raise ValueError(f"{source}/{split}: need {count} eligible prompts, found {len(ranked)}")
    return [entry[2] for entry in sorted(ranked)[:count]]


def materialize(root, source_roots, hf, tokenizer, *, seed=17, local=False):
    if root.exists() or set(source_roots) != set(SOURCES):
        raise ValueError("Use a new mixture directory and exactly gsm8k/math/ifeval prepared sources")
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise ValueError("Mixture requires an explicit checkpoint chat template")
    template_hash = digest(tokenizer.chat_template.encode())
    max_tokens = 480 if local else 2048
    selected = {split: {} for split in ("train", "eval")}
    provenance = {}
    for source in SOURCES:
        partitions, provenance[source] = read_source(Path(source_roots[source]), source, template_hash)
        for split, count in (("train", UPDATES * PROMPTS_PER_SOURCE), ("eval", PROMPTS_PER_SOURCE)):
            selected[split][source] = select_rows(
                partitions[split],
                source=source,
                split=split,
                count=count,
                seed=seed,
                tokenizer=tokenizer,
                max_prompt_tokens=max_tokens,
            )
    training = [
        row
        for update in range(UPDATES)
        for source in SOURCES
        for row in selected["train"][source][update * PROMPTS_PER_SOURCE : (update + 1) * PROMPTS_PER_SOURCE]
    ]
    evaluation = [row for source in SOURCES for row in selected["eval"][source]]
    if {row["input"] for row in training} & {row["input"] for row in evaluation}:
        raise ValueError("Selected training and held-out prompt text overlaps across sources")
    all_ids = [row["metadata"]["mixture"]["id"] for row in training + evaluation]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Selected source identities overlap across train and evaluation")
    root.mkdir(parents=True)
    hashes = {}
    for split, rows in (("train", training), ("eval", evaluation)):
        raw = b"".join(encoded(row) for row in rows)
        (root / f"{split}.jsonl").write_bytes(raw)
        hashes[split] = digest(raw)
    report = {
        "schema_version": 1,
        "task": "mixture",
        "profile": "local" if local else "sft",
        "hf": str(hf.resolve()),
        "seed": seed,
        "template_sha256": template_hash,
        "sources": provenance,
        "source_order": list(SOURCES),
        "prepared_sha256": hashes,
        "updates": UPDATES,
        "prompts_per_source_per_update": PROMPTS_PER_SOURCE,
        "samples_per_prompt": SAMPLES_PER_PROMPT,
        "schedule": [
            [row["metadata"]["mixture"]["id"] for row in training[start : start + BATCH_PROMPTS]]
            for start in range(0, UPDATES * BATCH_PROMPTS, BATCH_PROMPTS)
        ],
        "selection": "Sort each source/split by SHA256(seed, source, split, original identity); take a fixed quota",
        "interpretation": "Held out from these updates only; source-conditioned qualification, not benchmark significance",
    }
    (root / "preparation.json").write_bytes(encoded(report))
    return report


def verify(root):
    report = json.loads((root / "preparation.json").read_text())
    if report.get("schema_version") != 1 or report.get("source_order") != list(SOURCES):
        raise ValueError("Unsupported mixture manifest")
    if set(report.get("prepared_sha256", {})) != {"train", "eval"}:
        raise ValueError("Mixture requires hashes for train and eval only")
    rows = {}
    for split, expected in report["prepared_sha256"].items():
        raw = (root / f"{split}.jsonl").read_bytes()
        if digest(raw) != expected:
            raise ValueError(f"Prepared mixture {split} changed")
        rows[split] = [json.loads(line) for line in raw.decode().splitlines()]
    if (
        set(rows) != {"train", "eval"}
        or len(rows["train"]) != UPDATES * BATCH_PROMPTS
        or len(rows["eval"]) != BATCH_PROMPTS
    ):
        raise ValueError("Mixture requires 12 training and 6 held-out prompts")
    schedule = [
        [row["metadata"]["mixture"]["id"] for row in rows["train"][start : start + BATCH_PROMPTS]]
        for start in range(0, UPDATES * BATCH_PROMPTS, BATCH_PROMPTS)
    ]
    if schedule != report["schedule"]:
        raise ValueError("Mixture schedule changed")
    return report, rows
