"""Verify extracted heldout generations against retained audits and render an offline reader."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

BACKENDS = ("core", "megatron")
STEPS = (0, 20, 40, 60, 80, 100)


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_by(rows, key, context):
    result = {}
    for row in rows:
        identity = key(row)
        require(isinstance(identity, str) and identity, f"{context}: missing string ID")
        require(identity not in result, f"{context}: duplicate ID {identity}")
        result[identity] = row
    return result


def verify_generations(data, audits, prepared_bytes, *, expected_count=128, steps=STEPS):
    """Fail closed before exposing any response as a verified comparison."""
    require(data.get("schema_version") == 1, "Unsupported extraction schema")
    require(sha256(prepared_bytes) == data["prepared_sha256"], "Prepared file SHA mismatch")
    prepared_rows = [json.loads(line) for line in prepared_bytes.decode().splitlines() if line.strip()]
    require(prepared_rows == data["prepared"], "Extracted prepared rows differ from original JSONL")
    prepared = unique_by(prepared_rows, lambda r: r["metadata"]["prepared_sample_id"], "prepared")
    require(len(prepared) == expected_count, "Prepared question count mismatch")
    require(len({r["input"] for r in prepared.values()}) == expected_count, "Duplicate prepared prompts")
    expected = {(backend, step) for backend in BACKENDS for step in steps}
    require(len(data["dumps"]) == len(expected), "Dump count mismatch")
    audit_entries = {}
    for backend in BACKENDS:
        audit = audits[backend]
        require(audit.get("valid") is True and audit.get("backend") == backend, f"{backend}: invalid audit")
        require(audit["prepared_sha256"]["eval"] == data["prepared_sha256"], f"{backend}: prepared SHA mismatch")
        for entry in audit["evaluation"]:
            key = (backend, entry["completed_steps"])
            require(key not in audit_entries, f"{key}: duplicate audit step")
            audit_entries[key] = entry
    require(set(audit_entries) == expected, "Audited evaluation coverage differs")
    by_step, hashes, proofs = {}, {}, []
    for dump in data["dumps"]:
        key = (dump["backend"], dump["completed_steps"])
        require(key in expected and key not in by_step, f"Unexpected or duplicate dump {key}")
        entry = audit_entries[key]
        require(entry["valid"] is True, f"{key}: failed audit entry")
        require(Path(dump["file"]).name == entry["file"], f"{key}: audited filename mismatch")
        require(dump["sha256"] == entry["sha256"], f"{key}: audited dump SHA mismatch")
        samples = unique_by(dump["samples"], lambda r: r["id"], str(key))
        checked = unique_by(entry["samples"], lambda r: r["id"], f"audit {key}")
        require(set(samples) == set(checked) == set(prepared), f"{key}: question IDs differ")
        for identity, sample in samples.items():
            row, proof = prepared[identity], checked[identity]
            require(sample["prompt"] == row["input"], f"{key}/{identity}: prompt differs")
            require(sample["label"] == row["label"], f"{key}/{identity}: label differs")
            require(sample["metadata"]["prepared_sample_id"] == identity, f"{key}/{identity}: metadata ID differs")
            require(isinstance(sample["response"], str), f"{key}/{identity}: non-string response")
            response_sha = sha256(sample["response"].encode())
            require(response_sha == proof["response_sha256"], f"{key}/{identity}: response SHA mismatch")
            require(sample["reward"] == proof["correct"], f"{key}/{identity}: reward mismatch")
            require(sample["response_length"] == proof["response_tokens"], f"{key}/{identity}: token count mismatch")
            require(sample["status"] in ("completed", "truncated"), f"{key}/{identity}: unsupported status")
            require((sample["status"] == "truncated") == proof["truncated"], f"{key}/{identity}: truncation mismatch")
            require(
                sample["weight_versions"] in ([entry["policy_version"]], [str(entry["policy_version"])]),
                f"{key}/{identity}: policy version mismatch",
            )
            token_hash = sample["prompt_tokens_sha256"]
            require(isinstance(token_hash, str) and re.fullmatch(r"[0-9a-f]{64}", token_hash), "Invalid token hash")
            require(hashes.setdefault(identity, token_hash) == token_hash, f"{key}/{identity}: prompt tokens differ")
        by_step[key] = samples
        proofs.append({"backend": key[0], "completed_steps": key[1], "file": dump["file"], "sha256": dump["sha256"]})
    groups = {name: [] for name in ("core_only", "megatron_only", "both", "neither")}
    rows = []
    for identity, row in prepared.items():
        core = by_step[("core", steps[-1])][identity]["reward"] == 1
        mega = by_step[("megatron", steps[-1])][identity]["reward"] == 1
        group = "both" if core and mega else "core_only" if core else "megatron_only" if mega else "neither"
        groups[group].append(identity)
        rows.append(
            {
                "id": identity,
                "question": row["metadata"].get("query", row["input"]),
                "prompt": row["input"],
                "label": row["label"],
                "final_group": group,
                "generations": {
                    backend: {
                        str(step): {
                            **by_step[(backend, step)][identity],
                            "response_sha256": sha256(by_step[(backend, step)][identity]["response"].encode()),
                        }
                        for step in steps
                    }
                    for backend in BACKENDS
                },
            }
        )
    report = {
        "valid": True,
        "questions": len(prepared),
        "dumps": len(proofs),
        "responses_verified": len(prepared) * len(proofs),
        "prepared_sha256": data["prepared_sha256"],
        "identical_ids_prompts_labels_and_prompt_tokens": True,
        "final_groups": groups,
        "dump_provenance": proofs,
        "verification_scope": "Extracted response content hashes and source dump hash attestations match the passing audits; original JSONL bytes also verified. Raw .pt files are not reloaded by this reader.",
    }
    return {"steps": list(steps), "rows": rows, "verification": report}, report


def render_reader(payload):
    template = Path(__file__).with_name("gsm8k_generation_reader.html").read_text()
    # A literal '<' must never appear in the embedded JSON script body, including '</script>'.
    serialized = json.dumps(payload, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    return template.replace("__VERIFIED_GENERATIONS_JSON__", serialized)


def write_csv(payload, path):
    fields = ["id", "target", "final_group", "completed_steps"]
    for backend in BACKENDS:
        fields += [f"{backend}_{field}" for field in ("correct", "response_tokens", "status", "response_sha256")]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in payload["rows"]:
            for step in payload["steps"]:
                record = {
                    "id": row["id"],
                    "target": row["label"],
                    "final_group": row["final_group"],
                    "completed_steps": step,
                }
                for backend in BACKENDS:
                    sample = row["generations"][backend][str(step)]
                    record.update(
                        {
                            f"{backend}_correct": int(sample["reward"]),
                            f"{backend}_response_tokens": sample["response_length"],
                            f"{backend}_status": sample["status"],
                            f"{backend}_response_sha256": sample["response_sha256"],
                        }
                    )
                writer.writerow(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generations", type=Path)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--audit-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.generations.read_text())
    audits = {
        backend: json.loads((args.audit_directory / f"{backend}-audit.json").read_text()) for backend in BACKENDS
    }
    payload, report = verify_generations(data, audits, args.prepared.read_bytes())
    report["extraction_sha256"] = sha256(args.generations.read_bytes())
    report["audit_sha256"] = {
        backend: sha256((args.audit_directory / f"{backend}-audit.json").read_bytes()) for backend in BACKENDS
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison-reader.html").write_text(render_reader(payload))
    write_csv(payload, args.output / "question-outcomes.csv")
    (args.output / "generation-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "valid": True,
                "questions": report["questions"],
                "responses_verified": report["responses_verified"],
                "reader": str(args.output / "comparison-reader.html"),
            }
        )
    )


if __name__ == "__main__":
    main()
