"""Standalone adapter tests: python -m unittest discover -s scripts/data/decontam."""

import argparse
import gzip
import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
from importlib import util
from pathlib import Path
from unittest import mock

SPEC = util.spec_from_file_location("decon_report_adapter", Path(__file__).with_name("report.py"))
report = util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)

FAKE = """#!{python}
import json, pathlib, sys
args = sys.argv[1:]
assert args[0] == "detect" and "--purify" not in args
assert args[args.index("--content-key")+1] == "text"
for flag in ["--index-answers", "--index-passages"]:
    assert args[args.index(flag)+1] == "false"
config = json.loads(pathlib.Path(args[args.index("--config")+1]).read_text())
training = pathlib.Path(args[args.index("--training-dir")+1])
reports = pathlib.Path(args[args.index("--report-output-dir")+1])
files = sorted(training.glob("*.jsonl"))
rows = [json.loads(line) for path in files for line in path.read_text().splitlines()]
assert all(set(row) == {{"text"}} for row in rows)
assert all("SECRET_ASSISTANT" not in row["text"] for row in rows)
assert all("SECRET_SOURCE_ID" not in row["text"] for row in rows)
candidates = []
for source, line in config.get("targets", []):
    row = json.loads(files[source].read_text().splitlines()[line])
    assert "PLANTED_LITERAL" in row["text"]
    candidates.append({{"training_file": files[source].name, "training_line": line,
                       "eval_key": "heldout-v1", "eval_instance_index": 3,
                       "training_overlap_text": row["text"], "eval_answer_text": "SECRET_ANSWER"}})
if config.get("bad_candidate"):
    candidates = [{{"training_file":"UNKNOWN_PRIVATE_PATH", "training_line":999,
                   "eval_key":"heldout-v1", "eval_instance_index":3}}]
(reports/"raw.jsonl").write_text("".join(json.dumps(row)+"\\n" for row in candidates))
print("PRIVATE_RAW_LOG /PRIVATE_PATH SECRET_SOURCE_ID")
print("Using evaluation directory from --evals-dir: " + args[args.index("--evals-dir")+1], file=sys.stderr)
if config.get("soft_failure"):
    print("Error: PRIVATE_RAW_ERROR", file=sys.stderr)
if config.get("stdout_error"):
    print("Error processing reference PRIVATE_PATH")
if config.get("tokenization_failure"):
    print("Tokenization failed PRIVATE_PATH", file=sys.stderr)
if not config.get("missing_table"):
    for label, value in [("Training lines",len(rows)-config.get("skip_training",0)),
                         ("Lines indexed",config.get("reference_rows",1)-config.get("skip_reference",0)),
                         ("Total lines examined",config.get("reference_rows",1)),
                         ("Contaminated matches",len(candidates)+config.get("extra_matches",0))]:
        print("│ " + label + "  " + str(value) + " │")
sys.exit(config.get("returncode",0))
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sample(prompt="ordinary prompt with enough words for a test"):
    return {
        "id": "SECRET_SOURCE_ID",
        "messages": [
            {"role": "system", "content": "system context"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "SECRET_ASSISTANT"},
        ],
    }


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / "fake_decon"
        self.binary.write_text(FAKE.format(python=sys.executable))
        os.chmod(self.binary, 0o700)
        self.config = self.root / "config.yaml"
        self.config.write_text("{}")
        self.references = self.root / "evals"
        self.references.mkdir()
        self.reference = self.references / "heldout.jsonl"
        self.reference.write_text(
            json.dumps(
                {
                    "eval_key": "heldout-v1",
                    "eval_instance_index": 3,
                    "doc_id": 1,
                    "question": "heldout test question with enough distinct words",
                }
            )
            + "\n"
        )
        self.manifest = self.root / "reference-pins.json"
        self.manifest.write_text(json.dumps({self.reference.name: sha(self.reference.read_bytes())}))
        self.source = self.root / "PRIVATE_SOURCE.jsonl"
        self.write_source([sample()])

    def write_source(self, rows, path=None):
        path = path or self.source
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def run_adapter(self, config=None, **overrides):
        self.config.write_text(json.dumps(config or {}))
        args = argparse.Namespace(
            source=[self.source],
            decon_binary=self.binary,
            decon_binary_sha256=sha(self.binary.read_bytes()),
            decon_repo_sha="a" * 40,
            config=self.config,
            config_sha256=sha(self.config.read_bytes()),
            evals_dir=self.references,
            reference_manifest=self.manifest,
            work_dir=self.root / "private-run",
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        receipt, code = report.run_audit(args)
        return receipt, code, args.work_dir

    def test_normalization_excludes_assistant_and_tools(self):
        row = {
            "messages": [
                {"role": "system", "content": "  re\u0301sume\r\ncontext  "},
                {"role": "assistant", "content": None},
                {"role": "tool", "content": "SECRET"},
                {"role": "user", "content": "  question  "},
            ]
        }
        self.assertEqual(report.normalized_prompt(row), "résume\ncontext\n\nquestion")

    def test_complete_without_candidates_is_still_report_only(self):
        receipt, code, work = self.run_adapter()
        self.assertEqual(code, 0)
        self.assertTrue(receipt["complete"])
        self.assertEqual(receipt["mode"], "report_only")
        self.assertEqual(receipt["review_status"], "needs_review")
        self.assertEqual(stat.S_IMODE(work.stat().st_mode), 0o700)
        self.assertEqual(receipt["coverage"]["missing_training_rows"], 0)
        self.assertEqual(receipt["adapter_sha256"], sha(Path(report.__file__).read_bytes()))
        self.assertEqual(
            receipt["effective_cli_overrides"],
            {"content_key": "text", "index_answers": False, "index_passages": False, "purify": False},
        )

    def test_planted_literal_maps_source_and_zero_based_row(self):
        self.write_source([sample(), sample("PLANTED_LITERAL prompt in row one")])
        second = self.root / "another-private-source.jsonl"
        self.write_source([sample("PLANTED_LITERAL second source row zero")], second)
        receipt, code, work = self.run_adapter({"targets": [[0, 1], [1, 0]]}, source=[self.source, second])
        self.assertEqual(code, 1)
        self.assertEqual(receipt["candidate_matches"], 2)
        rows = [json.loads(line) for line in (work / "candidates.jsonl").read_text().splitlines()]
        self.assertEqual([(row["source_ordinal"], row["source_row"]) for row in rows], [(0, 1), (1, 0)])
        self.assertEqual(rows[0]["sha256"], sha(self.source.read_bytes().splitlines(keepends=True)[1]))
        self.assertEqual(
            rows[0]["normalized_text_sha256"],
            sha(report.normalized_prompt(sample("PLANTED_LITERAL prompt in row one")).encode()),
        )
        self.assertEqual(receipt["sources"][0]["original_sha256"], sha(self.source.read_bytes()))

    def test_receipts_never_copy_private_text_paths_or_source_ids(self):
        self.write_source([sample("PLANTED_LITERAL SECRET_USER_PROMPT")])
        receipt, _, work = self.run_adapter({"targets": [[0, 0]]})
        public = json.dumps(receipt) + (work / "candidates.jsonl").read_text()
        for secret in (
            "SECRET_USER_PROMPT",
            "PRIVATE_SOURCE",
            str(self.root),
            "SECRET_SOURCE_ID",
            "PRIVATE_RAW_LOG",
            "SECRET_ASSISTANT",
            "SECRET_ANSWER",
        ):
            self.assertNotIn(secret, public)
        self.assertIn("SECRET_USER_PROMPT", (work / "reports" / "raw.jsonl").read_text())
        self.assertNotIn("SECRET_ASSISTANT", (work / "training" / "source000000.jsonl").read_text())

    def test_nonzero_and_soft_failures(self):
        for index, config in enumerate(
            ({"returncode": 7}, {"soft_failure": True}, {"stdout_error": True}, {"tokenization_failure": True})
        ):
            with self.subTest(config=config):
                receipt, code, _ = self.run_adapter(config, work_dir=self.root / f"failure{index}")
                self.assertEqual(code, 2)
                self.assertFalse(receipt["complete"])

    def test_error_word_in_a_workdir_path_is_not_a_matcher_error(self):
        parent = self.root / "test-error"
        parent.mkdir()
        receipt, code, _ = self.run_adapter(work_dir=parent / "run")
        self.assertEqual(code, 0)
        self.assertTrue(receipt["complete"])
        self.assertNotIn("matcher_reported_error", receipt["issues"])

    def test_skip_training_count_fails_closed(self):
        receipt, code, _ = self.run_adapter({"skip_training": 1})
        self.assertEqual(code, 2)
        self.assertEqual(receipt["coverage"]["missing_training_rows"], 1)
        self.assertIn("incomplete_training_coverage", receipt["issues"])

    def test_reference_index_skip_fails_closed(self):
        receipt, code, _ = self.run_adapter({"skip_reference": 1})
        self.assertEqual(code, 2)
        self.assertEqual(receipt["coverage"]["missing_reference_rows"], 1)

    def test_missing_table_and_candidate_count_mismatch(self):
        for index, config in enumerate(({"missing_table": True}, {"extra_matches": 1}, {"bad_candidate": True})):
            with self.subTest(config=config):
                receipt, code, _ = self.run_adapter(config, work_dir=self.root / f"table{index}")
                self.assertEqual(code, 2)
                self.assertFalse(receipt["complete"])

    def test_malformed_and_empty_source_rows_are_not_skipped(self):
        for index, payload in enumerate(
            (
                b"",
                b"\n",
                b"not json\n",
                b"{}\n",
                json.dumps({"messages": [{"role": "assistant", "content": "answer"}]}).encode(),
                json.dumps({"messages": [{"role": "user", "content": []}]}).encode(),
                json.dumps({"messages": [{"role": ["user"], "content": "invalid role"}]}).encode(),
                b"[" * 2000 + b"]" * 2000,
            )
        ):
            with self.subTest(index=index):
                self.source.write_bytes(payload)
                receipt, code, work = self.run_adapter(work_dir=self.root / f"invalid{index}")
                self.assertEqual(code, 2)
                self.assertFalse(receipt["complete"])
                self.assertFalse((work / "stdout.log").exists())

    def test_pins_and_reference_inventory_are_required(self):
        for index, overrides in enumerate(
            ({"decon_repo_sha": "short"}, {"config_sha256": "0" * 64}, {"decon_binary_sha256": "0" * 64})
        ):
            with self.subTest(overrides=overrides):
                receipt, code, _ = self.run_adapter(work_dir=self.root / f"pin{index}", **overrides)
                self.assertEqual(code, 2)
                self.assertFalse(receipt["complete"])
        self.manifest.write_text("{}")
        self.assertEqual(self.run_adapter(work_dir=self.root / "missing-ref")[1], 2)

    def test_missing_invalid_or_duplicate_doc_ids_fail_before_matching(self):
        original = json.loads(self.reference.read_text())
        for index, document_id in enumerate((None, True, -1, 2**32 - 1, 2**32)):
            row = dict(original, doc_id=document_id)
            self.reference.write_text(json.dumps(row) + "\n")
            self.manifest.write_text(json.dumps({self.reference.name: sha(self.reference.read_bytes())}))
            receipt, code, work = self.run_adapter(work_dir=self.root / f"docid{index}")
            self.assertEqual(code, 2)
            self.assertIn("invalid_reference_document_id", receipt["issues"])
            self.assertFalse((work / "stdout.log").exists())
        other = dict(original, eval_instance_index=4)
        self.reference.write_text(json.dumps(original) + "\n" + json.dumps(other) + "\n")
        self.manifest.write_text(json.dumps({self.reference.name: sha(self.reference.read_bytes())}))
        receipt, code, work = self.run_adapter(work_dir=self.root / "duplicate-docid")
        self.assertEqual(code, 2)
        self.assertIn("duplicate_reference_document_id", receipt["issues"])
        self.assertFalse((work / "stdout.log").exists())

    def test_out_of_range_reference_instance_index_fails_before_matching(self):
        row = json.loads(self.reference.read_text())
        row["eval_instance_index"] = 2**64
        self.reference.write_text(json.dumps(row) + "\n")
        self.manifest.write_text(json.dumps({self.reference.name: sha(self.reference.read_bytes())}))
        receipt, code, work = self.run_adapter()
        self.assertEqual(code, 2)
        self.assertIn("invalid_reference_row", receipt["issues"])
        self.assertFalse((work / "stdout.log").exists())

    def test_literal_negative_zero_reference_ids_fail_before_matching(self):
        original = self.reference.read_text()
        for index, (field, value) in enumerate((("doc_id", 1), ("eval_instance_index", 3))):
            payload = original.replace(f'"{field}": {value}', f'"{field}": -0')
            self.assertNotEqual(original, payload)
            self.reference.write_text(payload)
            self.manifest.write_text(json.dumps({self.reference.name: sha(self.reference.read_bytes())}))
            receipt, code, work = self.run_adapter(work_dir=self.root / f"negative-zero{index}")
            self.assertEqual(code, 2)
            self.assertIn("noncanonical_json_integer", receipt["issues"])
            self.assertFalse((work / "stdout.log").exists())

    def test_valid_hash_truncated_gzip_is_a_sanitized_incomplete_run(self):
        compressed = gzip.compress(self.reference.read_bytes())[:-5]
        self.reference.unlink()
        path = self.references / "heldout.jsonl.gz"
        path.write_bytes(compressed)
        self.manifest.write_text(json.dumps({path.name: sha(compressed)}))
        receipt, code, work = self.run_adapter()
        self.assertEqual(code, 2)
        self.assertIn("input_or_execution_failure", receipt["issues"])
        self.assertFalse(receipt["complete"])
        self.assertFalse((work / "stdout.log").exists())
        self.assertNotIn(str(self.root), json.dumps(receipt))

    def test_existing_workdir_and_git_output_are_rejected(self):
        existing = self.root / "existing"
        existing.mkdir()
        (existing / "keep").write_text("kept")
        self.assertEqual(self.run_adapter(work_dir=existing)[1], 2)
        self.assertEqual((existing / "keep").read_text(), "kept")
        (self.root / ".git").mkdir()
        self.assertEqual(self.run_adapter(work_dir=self.root / "in-git")[1], 2)
        self.assertFalse((self.root / "in-git").exists())

    def test_receipt_io_failure_is_sanitized_and_restores_umask(self):
        original_write = Path.write_text

        def fail_receipt(path, *args, **kwargs):
            if path.name == "receipt.json":
                raise OSError("PRIVATE_PATH SECRET_ERROR")
            return original_write(path, *args, **kwargs)

        previous = os.umask(0o027)
        try:
            with mock.patch.object(Path, "write_text", fail_receipt):
                receipt, code, _ = self.run_adapter()
            self.assertEqual(code, 2)
            self.assertIn("receipt_write_failure", receipt["issues"])
            self.assertNotIn("PRIVATE_PATH", json.dumps(receipt))
            self.assertEqual(os.umask(0o027), 0o027)
        finally:
            os.umask(previous)

    def test_table_parser_rejects_duplicate_or_malformed_counts(self):
        for text in ("│ Training lines 1 │\n" * 2, "│ Training lines 1,2 │\n"):
            with self.assertRaises(report.AuditError):
                report.table_counts(text)


if __name__ == "__main__":
    unittest.main()
