#!/usr/bin/env python3
"""Run a pinned decon executable and produce text-free, report-only receipts."""

import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import unicodedata
from pathlib import Path

# decon 0.3 defaults (src/common/detection_config.rs, src/common/text.rs) for the settings mirrored here.
DEFAULT_NGRAM_SIZE = 5
DEFAULT_PUNCTUATION = "!'\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\u201c\u201d\u2018\u2019\u2014"
# A run of characters outside Unicode White_Space, which decon's Rust `\s` uses. Python's `\s` and
# str.split() also break on U+001C..U+001F, so they would overcount words.
MATCHER_WORD = re.compile(r"[^\t\n\x0b\x0c\r \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+")
CONFIG_LINE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*):(?:[ \t]+(.*))?")
MATCHER_ERROR = re.compile(
    r"^\s*(?:error\b|tokenization failed\b|thread '[^'\n]*' panicked\b)", re.IGNORECASE | re.MULTILINE
)
COUNT_LABELS = {
    "training_lines": "Training lines",
    "reference_lines_examined": "Total lines examined",
    "reference_lines_indexed": "Lines indexed",
    "contaminated_matches": "Contaminated matches",
}
# decon prints these index-summary rows only when they are nonzero.
SKIP_LABELS = {
    "reference_lines_skipped": "Lines skipped",
    "reference_duplicates": "- Duplicates",
    "reference_below_min_tokens": "- Below minimum tokens",
    "reference_below_min_unique_words": "- Below min unique words",
}
REFERENCE_TEXT_FIELDS = ("question", "answer", "passage")


class AuditError(Exception):
    """A fixed, text-free failure code safe to include in a receipt."""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check_hex(value, length):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value) is None:
        raise AuditError("invalid_pin")
    return value


def normalized_text(value):
    """Apply the newline and NFC normalization shared by training and reference text."""
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = unicodedata.normalize("NFC", value)
    return normalized, normalized != value


def normalized_prompt(row):
    """Return the joined system/user prompt and whether NFC normalization changed it."""
    if not isinstance(row, dict) or not isinstance(row.get("messages"), list) or not row["messages"]:
        raise AuditError("invalid_messages")
    parts = []
    has_user = non_nfc = False
    for message in row["messages"]:
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or message["role"] not in {"system", "user", "assistant", "tool", "environment"}
        ):
            raise AuditError("invalid_message_role")
        if message["role"] not in {"system", "user"}:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            raise AuditError("invalid_prompt_content")
        content, changed = normalized_text(content)
        content = content.strip()
        if content:
            parts.append(content)
            has_user |= message["role"] == "user"
            non_nfc |= changed
    if not has_user:
        raise AuditError("empty_user_prompt")
    return "\n\n".join(parts), non_nfc


def json_integer(token):
    if token == "-0":
        raise AuditError("noncanonical_json_integer")
    return int(token)


def json_line(raw):
    try:
        return json.loads(raw.decode("utf-8"), parse_int=json_integer)
    except (UnicodeError, ValueError):
        raise AuditError("invalid_jsonl") from None


def yaml_string(raw):
    double = re.fullmatch(r'"((?:[^"\\]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*)"(?:[ \t]+#.*)?', raw)
    if double is not None:
        return json.loads(f'"{double.group(1)}"', strict=False)
    single = re.fullmatch(r"'((?:[^']|'')*)'(?:[ \t]+#.*)?", raw)
    if single is not None:
        return single.group(1).replace("''", "'")
    raise AuditError("unsupported_config_syntax")


def config_settings(payload):
    """Read ngram_size and punctuation_chars from decon's flat YAML config, rejecting syntax not decoded here."""
    try:
        lines = re.split(r"\r\n|\r|\n", payload.decode("utf-8"))
    except UnicodeError:
        raise AuditError("unsupported_config_syntax") from None
    values = {}
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = CONFIG_LINE.fullmatch(line.rstrip())
        if match is None or match.group(1) in values:
            raise AuditError("unsupported_config_syntax")
        value = (match.group(2) or "").strip()
        if value.startswith(('"', "'")):
            yaml_string(value)  # Quoted scalars must close on their own line.
        values[match.group(1)] = value
    ngram_size = DEFAULT_NGRAM_SIZE
    if "ngram_size" in values:
        number = re.fullmatch(r"([1-9][0-9]{0,5})(?:[ \t]+#.*)?", values["ngram_size"])
        if number is None:
            raise AuditError("unsupported_config_syntax")
        ngram_size = int(number.group(1))
    punctuation = DEFAULT_PUNCTUATION
    if "punctuation_chars" in values:
        punctuation = yaml_string(values["punctuation_chars"])
    return ngram_size, str.maketrans(dict.fromkeys(punctuation, " "))


def matcher_words(text, punctuation_table):
    """Count words as decon's clean_text leaves them. Each such word becomes at least one matcher token."""
    return len(MATCHER_WORD.findall(text.lower().translate(punctuation_table)))


def stage_sources(paths, training, mapping, min_tokens, punctuation_table):
    sources = []
    names = {}
    mapping.execute(
        "CREATE TABLE rows (source INTEGER, row INTEGER, raw_sha TEXT, text_sha TEXT, PRIMARY KEY(source,row))"
    )
    for ordinal, path in enumerate(paths):
        if path.is_symlink() or not path.is_file():
            raise AuditError("invalid_source_file")
        name = f"source{ordinal:06d}.jsonl"
        names[name] = ordinal
        original_hash = hashlib.sha256()
        staged_hash = hashlib.sha256()
        rows = non_nfc_rows = possibly_short_rows = 0
        with path.open("rb") as source, (training / name).open("wb") as output:
            for row_number, raw in enumerate(source):
                original_hash.update(raw)
                text, non_nfc = normalized_prompt(json_line(raw))
                encoded = (json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8")
                output.write(encoded)
                staged_hash.update(encoded)
                mapping.execute(
                    "INSERT INTO rows VALUES (?,?,?,?)", (ordinal, row_number, digest(raw), digest(text.encode()))
                )
                rows += 1
                non_nfc_rows += non_nfc
                possibly_short_rows += matcher_words(text, punctuation_table) < min_tokens
        if not rows:
            raise AuditError("empty_source_file")
        sources.append(
            {
                "source_ordinal": ordinal,
                "original_sha256": original_hash.hexdigest(),
                "normalized_jsonl_sha256": staged_hash.hexdigest(),
                "rows": rows,
                "non_nfc_rows": non_nfc_rows,
                "rows_possibly_below_min_tokens": possibly_short_rows,
            }
        )
    mapping.commit()
    if not sources:
        raise AuditError("missing_sources")
    return sources, names


def stage_references(evals_dir, manifest_path, destination):
    if evals_dir.is_symlink() or not evals_dir.is_dir():
        raise AuditError("invalid_reference_directory")
    manifest = json_line(manifest_path.read_bytes())
    if not isinstance(manifest, dict) or not manifest:
        raise AuditError("invalid_reference_manifest")
    files = sorted(path for path in evals_dir.rglob("*") if path.is_file() or path.is_symlink())
    actual = {path.relative_to(evals_dir).as_posix(): path for path in files}
    if set(actual) != set(manifest):
        raise AuditError("reference_manifest_mismatch")
    references = []
    identities = set()
    document_ids = set()
    for ordinal, (relative, path) in enumerate(actual.items()):
        expected = check_hex(manifest[relative], 64)
        if path.is_symlink() or not (relative.endswith(".jsonl") or relative.endswith(".jsonl.gz")):
            raise AuditError("unsupported_reference_file")
        payload = path.read_bytes()
        if digest(payload) != expected:
            raise AuditError("reference_hash_mismatch")
        data = gzip.decompress(payload) if relative.endswith(".gz") else payload
        # decon reads lines split on LF only, like file iteration; bytes.splitlines would also split on CR.
        lines = data.split(b"\n")
        if lines[-1] == b"":
            lines.pop()
        staged = bytearray()
        rows = non_nfc_rows = 0
        for raw in lines:
            row = json_line(raw)
            if not isinstance(row, dict):
                raise AuditError("invalid_reference_row")
            key, index, question = row.get("eval_key"), row.get("eval_instance_index"), row.get("question")
            if (
                not isinstance(key, str)
                or not key
                or type(index) is not int
                or not 0 <= index <= 0xFFFFFFFFFFFFFFFF
                or not isinstance(question, str)
                or not question.strip()
            ):
                raise AuditError("invalid_reference_row")
            document_id = row.get("doc_id")
            if type(document_id) is not int or not 0 <= document_id < 0xFFFFFFFF:
                raise AuditError("invalid_reference_document_id")
            if document_id in document_ids:
                raise AuditError("duplicate_reference_document_id")
            document_ids.add(document_id)
            if (key, index) in identities:
                raise AuditError("ambiguous_reference_identity")
            identities.add((key, index))
            non_nfc = False
            for field in REFERENCE_TEXT_FIELDS:
                if isinstance(row.get(field), str):
                    row[field], changed = normalized_text(row[field])
                    non_nfc |= changed
            staged += (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            rows += 1
            non_nfc_rows += non_nfc
        if not rows:
            raise AuditError("empty_reference_file")
        (destination / f"reference{ordinal:06d}.jsonl").write_bytes(staged)
        references.append(
            {
                "reference_ordinal": ordinal,
                "sha256": expected,
                "staged_sha256": digest(staged),
                "rows": rows,
                "non_nfc_rows": non_nfc_rows,
            }
        )
    return references, identities


def table_counts(stdout):
    counts = {}
    for key, label in {**COUNT_LABELS, **SKIP_LABELS}.items():
        values = re.findall(r"^\s*│\s*" + re.escape(label) + r"\s+([0-9][0-9,]*)\s*│\s*$", stdout, re.MULTILINE)
        if not values and key in SKIP_LABELS:
            counts[key] = 0
            continue
        if len(values) != 1 or re.fullmatch(r"(?:[0-9]+|[1-9][0-9]{0,2}(?:,[0-9]{3})+)", values[0]) is None:
            raise AuditError("missing_or_ambiguous_matcher_table")
        counts[key] = int(values[0].replace(",", ""))
    return counts


def sanitized_candidates(report_dir, names, mapping, identities):
    """Return one text-free row per (training row, eval item), counting decon's matches at different offsets."""
    positions = {}
    for path in sorted(report_dir.rglob("*.jsonl")):
        if path.is_symlink():
            raise AuditError("invalid_report_file")
        with path.open("rb") as stream:
            for raw in stream:
                row = json_line(raw)
                if not isinstance(row, dict):
                    raise AuditError("invalid_matcher_report")
                name, line = row.get("training_file"), row.get("training_line")
                key, index = row.get("eval_key"), row.get("eval_instance_index")
                if (
                    not isinstance(name, str)
                    or name not in names
                    or type(line) is not int
                    or line < 0
                    or not isinstance(key, str)
                    or type(index) is not int
                    or (key, index) not in identities
                ):
                    raise AuditError("unmapped_matcher_candidate")
                pair = (names[name], line, key, index)
                positions[pair] = positions.get(pair, 0) + 1
    candidates = []
    for (source, line, key, index), count in sorted(positions.items()):
        hashes = mapping.execute(
            "SELECT raw_sha,text_sha FROM rows WHERE source=? AND row=?", (source, line)
        ).fetchone()
        if hashes is None:
            raise AuditError("unmapped_matcher_candidate")
        candidates.append(
            {
                "source_ordinal": source,
                "source_row": line,
                "sha256": hashes[0],
                "normalized_text_sha256": hashes[1],
                "eval_key": key,
                "eval_instance_index": index,
                "positions": count,
            }
        )
    return candidates


def private_workdir(path):
    path = path.expanduser().absolute()
    if path.is_symlink() or any((parent / ".git").exists() for parent in [path, *path.resolve().parents]):
        raise AuditError("workdir_must_be_fresh_outside_git")
    path.mkdir(mode=0o700)
    os.chmod(path, 0o700)
    return path


def run_audit(args):
    receipt = {
        "schema_version": 2,
        "mode": "report_only",
        "review_status": "needs_review",
        "complete": False,
        "issues": [],
        "sources": [],
        "references": [],
        "candidate_pairs": 0,
        "candidate_matches": 0,
        "coverage": {
            "expected_training_rows": None,
            "processed_training_rows": None,
            "missing_training_rows": None,
            "expected_reference_rows": None,
            "examined_reference_rows": None,
            "indexed_reference_rows": None,
            "missing_reference_rows": None,
        },
        # Rows decon skips by design. They were not checked for overlap and need separate review.
        "coverage_limits": {
            "training_rows_below_min_tokens": None,
            "training_rows_possibly_below_min_tokens": None,
            "reference_rows_below_min_tokens": None,
            "reference_rows_below_min_unique_words": None,
            "reference_rows_duplicate": None,
        },
    }
    work = None
    candidates = []
    old_umask = os.umask(0o077)
    try:
        work = private_workdir(args.work_dir)
        receipt["adapter_sha256"] = digest(Path(__file__).read_bytes())
        receipt["effective_cli_overrides"] = {
            "content_key": "text",
            "index_answers": False,
            "index_passages": False,
            "purify": False,
        }
        training, references, reports = (work / name for name in ("training", "references", "reports"))
        for path in (training, references, reports):
            path.mkdir(mode=0o700)
        receipt["pins"] = {
            "decon_binary_sha256": check_hex(args.decon_binary_sha256, 64),
            "decon_repository_commit": check_hex(args.decon_repo_sha, 40),
            "config_sha256": check_hex(args.config_sha256, 64),
        }
        payloads = {}
        for name, source, expected in (
            ("decon", args.decon_binary, args.decon_binary_sha256),
            ("config.yaml", args.config, args.config_sha256),
        ):
            payloads[name] = source.read_bytes()
            if digest(payloads[name]) != expected:
                raise AuditError("binary_or_config_hash_mismatch")
            (work / name).write_bytes(payloads[name])
        os.chmod(work / "decon", 0o700)
        ngram_size, punctuation_table = config_settings(payloads["config.yaml"])
        # decon skips training lines with fewer than 2 * ngram_size tokens.
        receipt["matcher_settings"] = {"ngram_size": ngram_size, "training_min_tokens": 2 * ngram_size}
        receipt["effective_cli_overrides"]["ngram_size"] = ngram_size
        with sqlite3.connect(work / "mapping.sqlite3") as mapping:
            receipt["sources"], names = stage_sources(
                args.source, training, mapping, 2 * ngram_size, punctuation_table
            )
            coverage = receipt["coverage"]
            limits = receipt["coverage_limits"]
            coverage["expected_training_rows"] = sum(source["rows"] for source in receipt["sources"])
            possibly_short = sum(source["rows_possibly_below_min_tokens"] for source in receipt["sources"])
            receipt["references"], identities = stage_references(args.evals_dir, args.reference_manifest, references)
            coverage["expected_reference_rows"] = sum(ref["rows"] for ref in receipt["references"])
            command = [
                str(work / "decon"),
                "detect",
                "--training-dir",
                str(training),
                "--evals-dir",
                str(references),
                "--content-key",
                "text",
                "--config",
                str(work / "config.yaml"),
                "--report-output-dir",
                str(reports),
                "--index-answers",
                "false",
                "--index-passages",
                "false",
                # Same value as the config; passing it guarantees the skip bound uses decon's setting.
                "--ngram-size",
                str(ngram_size),
            ]
            with (work / "stdout.log").open("wb") as stdout, (work / "stderr.log").open("wb") as stderr:
                result = subprocess.run(
                    command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, cwd=work, check=False
                )
            stdout = (work / "stdout.log").read_text(encoding="utf-8", errors="replace")
            stderr = (work / "stderr.log").read_text(encoding="utf-8", errors="replace")
            receipt["matcher_returncode"] = result.returncode
            receipt["private_log_sha256"] = {
                "stdout": digest((work / "stdout.log").read_bytes()),
                "stderr": digest((work / "stderr.log").read_bytes()),
            }
            if result.returncode:
                receipt["issues"].append("matcher_nonzero_exit")
            if MATCHER_ERROR.search(stdout + "\n" + stderr):
                receipt["issues"].append("matcher_reported_error")
            counts = table_counts(stdout)
            receipt["matcher_counts"] = counts
            coverage.update(
                {
                    "processed_training_rows": counts["training_lines"],
                    "missing_training_rows": max(0, coverage["expected_training_rows"] - counts["training_lines"]),
                    "examined_reference_rows": counts["reference_lines_examined"],
                    "indexed_reference_rows": counts["reference_lines_indexed"],
                    "missing_reference_rows": max(
                        0, coverage["expected_reference_rows"] - counts["reference_lines_indexed"]
                    ),
                }
            )
            # decon reports no count for short training lines. Attribute the gap to that rule only when it
            # fits within the rows that could be that short; a longer row is never skipped by it.
            training_gap = coverage["expected_training_rows"] - counts["training_lines"]
            limits["training_rows_possibly_below_min_tokens"] = possibly_short
            if 0 <= training_gap <= possibly_short:
                limits["training_rows_below_min_tokens"] = training_gap
            else:
                receipt["issues"].append("incomplete_training_coverage")
            reference_skips = (
                counts["reference_duplicates"]
                + counts["reference_below_min_tokens"]
                + counts["reference_below_min_unique_words"]
            )
            if (
                counts["reference_lines_examined"] != coverage["expected_reference_rows"]
                or counts["reference_lines_indexed"] + reference_skips != coverage["expected_reference_rows"]
                or counts["reference_lines_skipped"] != reference_skips
            ):
                receipt["issues"].append("incomplete_reference_coverage")
            else:
                limits["reference_rows_below_min_tokens"] = counts["reference_below_min_tokens"]
                limits["reference_rows_below_min_unique_words"] = counts["reference_below_min_unique_words"]
                limits["reference_rows_duplicate"] = counts["reference_duplicates"]
            candidates = sanitized_candidates(reports, names, mapping, identities)
            receipt["candidate_pairs"] = len(candidates)
            receipt["candidate_matches"] = sum(row["positions"] for row in candidates)
            if receipt["candidate_matches"] != counts["contaminated_matches"]:
                receipt["issues"].append("candidate_count_mismatch")
        receipt["complete"] = not receipt["issues"]
    except AuditError as exc:
        receipt["issues"].append(str(exc))
    except Exception:
        receipt["issues"].append("input_or_execution_failure")
    finally:
        try:
            if work is not None:
                candidate_bytes = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in candidates)
                (work / "candidates.jsonl").write_bytes(candidate_bytes)
                receipt["candidates_sha256"] = digest(candidate_bytes)
                (work / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        except Exception:
            receipt["complete"] = False
            receipt["issues"].append("receipt_write_failure")
        finally:
            os.umask(old_umask)
    return receipt, 2 if not receipt["complete"] else (1 if candidates else 0)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--source", type=Path, action="append", required=True, help="Uncompressed messages JSONL; repeatable"
    )
    result.add_argument("--decon-binary", type=Path, required=True)
    result.add_argument("--decon-binary-sha256", required=True)
    result.add_argument("--decon-repo-sha", required=True, help="Full source repository commit asserted by the caller")
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--config-sha256", required=True)
    result.add_argument("--evals-dir", type=Path, required=True)
    result.add_argument(
        "--reference-manifest", type=Path, required=True, help="JSON object mapping relative reference files to SHA256"
    )
    result.add_argument(
        "--work-dir", type=Path, required=True, help="New private directory outside Git; must not already exist"
    )
    return result


def main():
    receipt, code = run_audit(parser().parse_args())
    print(json.dumps(receipt, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
