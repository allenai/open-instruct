from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from viewer.build_browsecomp_plus_urls import build_mapping
from viewer.evaluation_store import EvaluationStore
from viewer.training_registry import TrainingRegistry, TrainingRegistryError

TRAINING_ID = "evaluation-training"
BENCHMARK = "browsecomp-plus-bm25-830"
STEP = 60
EVALUATION_ID = f"{BENCHMARK}-step-{STEP}"


def _response(label: str, *, tool_name: str = "search", tool_arguments: str = '{"query": "alpha"}') -> dict[str, Any]:
    return {
        "finish_reason": "stop",
        "response": f"{label} answer",
        "tool_call_counts": {tool_name: 1},
        "rollout_states": {"termination_reason": "generation_complete"},
        "result": [
            {"role": "assistant", "reasoning_content": f"{label} reasoning"},
            {"role": "assistant", "function_call": {"name": tool_name, "arguments": tool_arguments}},
            {"role": "function", "name": tool_name, "content": f"{label} tool output"},
            {"role": "assistant", "content": f"{label} answer"},
        ],
    }


def _raw_record(*responses: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "Qwen3.5-9B",
        "prompt": {"formatted_prompt": "Question: identify Alpha"},
        "responses": list(responses),
    }


def _evaluation_row(
    query_id: str,
    *,
    correct: bool | None,
    completed: bool = True,
    response: str = "Alpha",
    question: str | None = None,
    json_path: str | None = None,
    parse_error: str | None = None,
    tool_calls: int = 1,
    step_count: int = 4,
) -> dict[str, Any]:
    judge_result: dict[str, Any] = {
        "correct": correct,
        "extracted_final_answer": response,
        "reasoning": f"Judge reasoning for {query_id}",
    }
    if parse_error is not None:
        judge_result["parse_error"] = parse_error
    return {
        "id": query_id,
        "query_id": query_id,
        "question": question or f"Question {query_id}",
        "response": response,
        "correct_answer": "Alpha",
        "is_completed": completed,
        "judge_prompt": f"Judge prompt for {query_id}",
        "judge_response": f"Judge output for {query_id}",
        "judge_result": judge_result,
        "json_path": json_path or f"{query_id}.json",
        "finish_reason": "stop" if completed else "length",
        "tool_call_counts": {"search": tool_calls},
        "step_count": step_count,
    }


def _write_registry(
    repo_root: Path,
    *,
    artifact_path: str,
    correct: int,
    total: int,
    judged_response_index: int = -1,
    benchmark: str = BENCHMARK,
) -> TrainingRegistry:
    registry_root = repo_root / "viewer" / "registry"
    trainings_root = registry_root / "trainings"
    trainings_root.mkdir(parents=True, exist_ok=True)
    config = {"schema_version": 1, "kind": "training_registry", "repo_root": "../..", "defaults": {}}
    training = {
        "schema_version": 1,
        "kind": "training",
        "id": TRAINING_ID,
        "title": "Evaluation training",
        "classification": "evaluated",
        "visibility": "default",
        "tags": {},
        "launches": [],
        "artifacts": {
            "evaluations": [
                {
                    "benchmark": benchmark,
                    "step": STEP,
                    "correct": correct,
                    "total": total,
                    "inference_artifact": {
                        "path": artifact_path,
                        "schema": "open_instruct_inference_v1",
                        "judged_response_index": judged_response_index,
                    },
                }
            ]
        },
    }
    (registry_root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (trainings_root / f"{TRAINING_ID}.yaml").write_text(yaml.safe_dump(training, sort_keys=False), encoding="utf-8")
    return TrainingRegistry(registry_root)


def _write_artifact(repo_root: Path, rows: list[dict[str, Any]], raw_records: dict[str, dict[str, Any]]) -> Path:
    artifact = repo_root / "output" / "evaluation-run"
    evaluation_dir = artifact / "eval"
    (evaluation_dir / "judge_results").mkdir(parents=True)
    (evaluation_dir / "evaluation_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (evaluation_dir / "evaluation_summary.json").write_text(
        json.dumps(
            {"num_results": len(rows), "num_correct": sum(row["judge_result"].get("correct") is True for row in rows)}
        ),
        encoding="utf-8",
    )
    for filename, payload in raw_records.items():
        (artifact / filename).write_text(json.dumps(payload), encoding="utf-8")
    return artifact


def _write_browsecomp_plus_urls(repo_root: Path, records: list[dict[str, Any]]) -> None:
    path = repo_root / "viewer" / "data" / "browsecomp_plus_urls.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


class EvaluationStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temporary.name)
        rows = [
            _evaluation_row("q1", correct=True, json_path="/stale/moved/run-q1.json", tool_calls=1, step_count=4),
            _evaluation_row("q2", correct=False, completed=False, response="Beta", tool_calls=2, step_count=8),
            _evaluation_row("q3", correct=None, parse_error="unparseable", response="Gamma"),
            _evaluation_row(
                "q4",
                correct=False,
                response="Delta",
                question="Question with a unique needle",
                tool_calls=5,
                step_count=12,
            ),
        ]
        self.artifact = _write_artifact(
            self.repo_root,
            rows,
            {
                "run-q1.json": _raw_record(_response("Only", tool_arguments='{"query": "beta"}')),
                "q2.json": _raw_record(_response("Incomplete")),
                "q3.json": _raw_record(_response("Unparsed")),
                "q4.json": _raw_record(
                    _response("First"),
                    _response("Second", tool_name="visit", tool_arguments='{ "url": "https://example.test" }'),
                ),
            },
        )
        self.registry = _write_registry(self.repo_root, artifact_path="output/evaluation-run", correct=1, total=4)
        self.store = EvaluationStore(self.registry)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_registry_exposes_complete_inference_artifact(self) -> None:
        evaluation = self.registry.get(TRAINING_ID).evaluations[0]
        self.assertEqual(evaluation.id, EVALUATION_ID)
        self.assertEqual(evaluation.inference_artifact.path, self.artifact)
        self.assertEqual(evaluation.inference_artifact.judged_response_index, -1)
        artifact = evaluation.public(self.repo_root)["inference_artifact"]
        self.assertEqual(artifact["display_path"], "output/evaluation-run")
        self.assertTrue(artifact["complete"])

    def test_query_reports_counts_and_filters_searches_and_sorts(self) -> None:
        result = self.store.query(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID)
        self.assertEqual(
            result["counts"],
            {
                "total": 4,
                "judged_correct": 1,
                "judged_incorrect": 1,
                "incomplete": 1,
                "accounted": 3,
                "unaccounted": 1,
            },
        )
        self.assertIn("account for 3 of the benchmark's 4 records", result["accounting_warning"])
        self.assertEqual(result["integrity_warnings"], [])
        self.assertEqual([row["query_id"] for row in result["records"]], ["q1", "q2", "q3", "q4"])
        self.assertEqual(
            [row["outcome"] for row in result["records"]],
            ["judged_correct", "incomplete", "unaccounted", "judged_incorrect"],
        )
        self.assertIsNone(result["records"][1]["correct"])

        incomplete = self.store.query(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, outcome="incomplete")
        self.assertEqual([row["query_id"] for row in incomplete["records"]], ["q2"])
        correct = self.store.query(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, outcome="judged_correct")
        self.assertEqual([row["query_id"] for row in correct["records"]], ["q1"])
        incorrect = self.store.query(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, outcome="judged_incorrect")
        self.assertEqual([row["query_id"] for row in incorrect["records"]], ["q4"])
        search = self.store.query(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, search="UNIQUE NEEDLE")
        self.assertEqual([row["query_id"] for row in search["records"]], ["q4"])
        by_tools = self.store.query(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, sort="tools")
        self.assertEqual(by_tools["records"][0]["query_id"], "q4")

    def test_complete_run_has_no_accounting_warning(self) -> None:
        rows = [
            _evaluation_row("q1", correct=True),
            _evaluation_row("q2", correct=False),
            _evaluation_row("q3", correct=False, completed=False),
        ]
        artifact = self.repo_root / "output" / "complete-evaluation-run"
        evaluation_dir = artifact / "eval"
        (evaluation_dir / "judge_results").mkdir(parents=True)
        (evaluation_dir / "evaluation_results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (evaluation_dir / "evaluation_summary.json").write_text(
            json.dumps({"num_results": 3, "num_correct": 1}), encoding="utf-8"
        )
        registry = _write_registry(self.repo_root, artifact_path="output/complete-evaluation-run", correct=1, total=3)
        result = EvaluationStore(registry).query(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID)
        self.assertEqual(result["counts"]["accounted"], 3)
        self.assertEqual(result["counts"]["unaccounted"], 0)
        self.assertIsNone(result["accounting_warning"])

    def test_one_and_multiple_response_selection_tracks_judge_applicability(self) -> None:
        single = self.store.detail(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="q1")
        self.assertEqual(single["selected_response_index"], 0)
        self.assertEqual(single["judged_response_index"], 0)
        self.assertTrue(single["judge_applies"])
        self.assertEqual(len(single["responses"]), 1)

        default = self.store.detail(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="q4")
        self.assertEqual(default["selected_response_index"], 1)
        self.assertEqual(default["judged_response_index"], 1)
        self.assertTrue(default["judge_applies"])
        self.assertTrue(default["judge"]["applies_to_selected_response"])
        self.assertEqual([choice["judged"] for choice in default["responses"]], [False, True])
        self.assertEqual(default["response"]["terminal_text"], "Second answer")

        first = self.store.detail(
            training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="q4", response_index=0
        )
        self.assertEqual(first["response"]["terminal_text"], "First answer")
        self.assertFalse(first["judge_applies"])
        self.assertFalse(first["judge"]["applies_to_selected_response"])

        registry = _write_registry(
            self.repo_root, artifact_path="output/evaluation-run", correct=1, total=4, judged_response_index=0
        )
        first_judged_store = EvaluationStore(registry)
        last_default = first_judged_store.detail(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="q4")
        self.assertEqual(last_default["selected_response_index"], 1)
        self.assertEqual(last_default["judged_response_index"], 0)
        self.assertFalse(last_default["judge_applies"])

    def test_detail_preserves_structured_turn_order(self) -> None:
        detail = self.store.detail(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="q4")
        self.assertEqual(
            [segment["kind"] for segment in detail["segments"]],
            ["prompt", "reasoning", "tool_call", "tool_result", "final_output"],
        )
        self.assertEqual(detail["segments"][2]["tool_name"], "visit")
        self.assertEqual(detail["segments"][3]["content"], "Second tool output")
        self.assertEqual(
            detail["kind_counts"], {"prompt": 1, "reasoning": 1, "tool_call": 1, "tool_result": 1, "final_output": 1}
        )

    def test_reference_answer_matches_only_trajectory_segments(self) -> None:
        detail = self.store.detail(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="q1")
        self.assertEqual(detail["reference_matches"]["term"], "Alpha")
        self.assertEqual(detail["reference_matches"]["total"], 0)
        self.assertEqual(
            detail["reference_matches"]["counts"],
            {"reasoning": 0, "tool_call": 0, "tool_result": 0, "final_output": 0},
        )
        self.assertEqual(detail["reference_matches"]["matches"], [])
        self.assertEqual(detail["segments"][0]["kind"], "prompt")
        self.assertIn("Alpha", detail["segments"][0]["content"])
        self.assertEqual(detail["segments"][0]["reference_answer_match_count"], 0)

        raw_path = self.artifact / "run-q1.json"
        raw_path.write_text(json.dumps(_raw_record(_response("ALPHA"))), encoding="utf-8")
        detail = self.store.detail(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="q1")
        reference_matches = detail["reference_matches"]
        self.assertEqual(reference_matches["term"], "Alpha")
        self.assertEqual(reference_matches["total"], 4)
        self.assertEqual(
            reference_matches["counts"], {"reasoning": 1, "tool_call": 1, "tool_result": 1, "final_output": 1}
        )
        self.assertEqual(
            [match["category"] for match in reference_matches["matches"]], list(reference_matches["counts"])
        )
        self.assertTrue(all(match["in_preview"] for match in reference_matches["matches"]))
        self.assertTrue(
            all(
                match["excerpt"][match["excerpt_match_start"] : match["excerpt_match_end"]].casefold() == "alpha"
                for match in reference_matches["matches"]
            )
        )
        self.assertEqual([segment["reference_answer_match_count"] for segment in detail["segments"]], [0, 1, 1, 1, 1])

    def test_browsecomp_plus_search_and_visit_results_are_annotated_from_evidence_urls(self) -> None:
        _write_browsecomp_plus_urls(
            self.repo_root,
            [
                {
                    "query_id": "q4",
                    "evidence_urls": ["https://example.test/evidence", "https://www.example.test/gold/"],
                    "positive_urls": ["https://www.example.test/gold/"],
                }
            ],
        )
        search_response = _response("Documents")
        search_response["result"][2]["content"] = (
            "**Gold result**\nGold excerpt\nSource: https://example.test/gold\n\n"
            "**Evidence result**\nEvidence excerpt\nSource: https://example.test/evidence"
        )
        (self.artifact / "q4.json").write_text(json.dumps(_raw_record(search_response)), encoding="utf-8")

        detail = self.store.detail(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="q4")
        relevance = detail["document_relevance"]
        self.assertTrue(relevance["available"])
        self.assertEqual(relevance["matched_evidence_url_count"], 2)
        self.assertEqual(relevance["matched_positive_url_count"], 1)
        self.assertEqual(relevance["evidence_result_count"], 1)
        self.assertEqual(relevance["positive_result_count"], 1)
        tool_result = next(segment for segment in detail["segments"] if segment["kind"] == "tool_result")
        self.assertEqual(tool_result["document_match_counts"], {"evidence": 2, "positive": 1, "evidence_only": 1})
        self.assertEqual([region["kind"] for region in tool_result["document_regions"]], ["gold", "evidence"])
        self.assertLess(tool_result["document_regions"][0]["end"], tool_result["document_regions"][1]["end"])

        visit_response = _response(
            "Visited", tool_name="visit", tool_arguments='{ "url": "http://example.test/gold" }'
        )
        visit_response["result"][2]["content"] = "The returned page body does not repeat its source URL."
        (self.artifact / "q4.json").write_text(json.dumps(_raw_record(visit_response)), encoding="utf-8")
        detail = self.store.detail(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="q4")
        tool_result = next(segment for segment in detail["segments"] if segment["kind"] == "tool_result")
        self.assertEqual(tool_result["document_match_counts"], {"evidence": 1, "positive": 1, "evidence_only": 0})
        self.assertEqual(tool_result["document_regions"][0]["kind"], "gold")
        self.assertEqual(tool_result["document_regions"][0]["start"], 0)
        self.assertEqual(tool_result["document_regions"][0]["end"], len(tool_result["content"]))

    def test_original_browsecomp_does_not_use_browsecomp_plus_url_annotations(self) -> None:
        registry = _write_registry(
            self.repo_root,
            artifact_path="output/evaluation-run",
            correct=1,
            total=4,
            benchmark="browsecomp-serper-jina-1266",
        )
        detail = EvaluationStore(registry).detail(
            training_id=TRAINING_ID, evaluation_id="browsecomp-serper-jina-1266-step-60", query_id="q4"
        )
        self.assertEqual(detail["document_relevance"], {"available": False, "browsecomp_plus": False})
        self.assertTrue(all("document_match_counts" not in segment for segment in detail["segments"]))

    def test_matches_defaults_to_last_response_and_can_select_another(self) -> None:
        default = self.store.matches(
            training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="q4", query="fIrSt"
        )
        self.assertEqual(default["selected_response_index"], 1)
        self.assertEqual(default["total"], 0)
        self.assertEqual(default["counts"], {"reasoning": 0, "tool_call": 0, "tool_result": 0, "final_output": 0})

        first = self.store.matches(
            training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="q4", query="fIrSt", response_index=0
        )
        self.assertEqual(first["selected_response_index"], 0)
        self.assertEqual(first["total"], 3)
        self.assertEqual(first["counts"], {"reasoning": 1, "tool_call": 0, "tool_result": 1, "final_output": 1})
        self.assertEqual(
            [match["category"] for match in first["matches"]], ["reasoning", "tool_result", "final_output"]
        )

    def test_matches_searches_full_segment_beyond_eight_thousand_character_preview(self) -> None:
        hidden_term = "needle-beyond-preview"
        response = _response("Long", tool_name="visit")
        response["result"][2]["content"] = "x" * 8_050 + hidden_term + " tail"
        (self.artifact / "q4.json").write_text(json.dumps(_raw_record(response)), encoding="utf-8")

        result = self.store.matches(
            training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="q4", query=hidden_term.upper()
        )
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["counts"], {"reasoning": 0, "tool_call": 0, "tool_result": 1, "final_output": 0})
        match = result["matches"][0]
        self.assertEqual(match["segment_kind"], "tool_result")
        self.assertGreater(match["start"], 8_000)
        self.assertFalse(match["in_preview"])
        self.assertEqual(
            match["excerpt"][match["excerpt_match_start"] : match["excerpt_match_end"]].casefold(), hidden_term
        )

    def test_segment_loads_complete_visit_content_and_full_document_region(self) -> None:
        _write_browsecomp_plus_urls(
            self.repo_root,
            [
                {
                    "query_id": "q4",
                    "evidence_urls": ["https://example.test/gold"],
                    "positive_urls": ["https://example.test/gold"],
                }
            ],
        )
        content = "x" * 8_050 + " Alpha beyond preview"
        response = _response("Visited", tool_name="visit", tool_arguments='{ "url": "https://example.test/gold" }')
        response["result"][2]["content"] = content
        (self.artifact / "q4.json").write_text(json.dumps(_raw_record(response)), encoding="utf-8")

        detail = self.store.detail(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="q4")
        preview = next(segment for segment in detail["segments"] if segment["kind"] == "tool_result")
        self.assertTrue(preview["truncated"])
        self.assertEqual(len(preview["content"]), 8_000)

        result = self.store.segment(
            training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="q4", segment_index=preview["index"]
        )
        segment = result["segment"]
        self.assertEqual(segment["content"], content)
        self.assertEqual(segment["char_len"], len(content))
        self.assertFalse(segment["truncated"])
        self.assertEqual(result["reference_matches"]["total"], 1)
        self.assertTrue(result["reference_matches"]["matches"][0]["in_preview"])
        self.assertEqual(segment["document_regions"][0]["kind"], "gold")
        self.assertEqual(segment["document_regions"][0]["start"], 0)
        self.assertEqual(segment["document_regions"][0]["end"], len(content))

    def test_matches_caps_occurrence_metadata_but_preserves_exact_counts(self) -> None:
        response = _response("Many", tool_name="visit")
        response["result"][2]["content"] = " ".join(["repeatme"] * 1_025)
        (self.artifact / "q4.json").write_text(json.dumps(_raw_record(response)), encoding="utf-8")

        result = self.store.matches(
            training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="q4", query="repeatme"
        )

        self.assertEqual(result["total"], 1_025)
        self.assertEqual(result["counts"]["tool_result"], 1_025)
        self.assertEqual(result["returned"], 1_000)
        self.assertEqual(len(result["matches"]), 1_000)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["limit"], 1_000)

    def test_matches_rejects_empty_and_oversized_queries(self) -> None:
        for query, message in (("   ", "must not be empty"), ("x" * 257, "at most 256 characters")):
            with self.subTest(query_length=len(query)), self.assertRaisesRegex(TrainingRegistryError, message):
                self.store.matches(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="q4", query=query)

    def test_invalid_response_index_is_rejected(self) -> None:
        for response_index in (-3, 2):
            with (
                self.subTest(response_index=response_index),
                self.assertRaisesRegex(TrainingRegistryError, "response index .* is outside"),
            ):
                self.store.detail(
                    training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="q4", response_index=response_index
                )


class EvaluationStoreArtifactSafetyTest(unittest.TestCase):
    def _store_with_row(
        self, repo_root: Path, row: dict[str, Any], *, registry_correct: int = 0, registry_total: int = 1
    ) -> EvaluationStore:
        _write_artifact(repo_root, [row], {})
        registry = _write_registry(
            repo_root, artifact_path="output/evaluation-run", correct=registry_correct, total=registry_total
        )
        return EvaluationStore(registry)

    def test_missing_raw_record_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self._store_with_row(Path(temporary), _evaluation_row("missing", correct=False))
            with self.assertRaisesRegex(TrainingRegistryError, "Could not locate raw inference JSON"):
                store.detail(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="missing")

    def test_out_of_artifact_json_path_is_not_opened(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            outside = repo_root / "elsewhere" / "outside-only.json"
            outside.parent.mkdir()
            outside.write_text(json.dumps(_raw_record(_response("External"))), encoding="utf-8")
            row = _evaluation_row("outside", correct=False, json_path=str(outside))
            store = self._store_with_row(repo_root, row)
            with self.assertRaisesRegex(TrainingRegistryError, "under .*evaluation-run"):
                store.detail(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID, query_id="outside")

    def test_registry_and_evaluation_artifact_count_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self._store_with_row(Path(temporary), _evaluation_row("only-row", correct=False), registry_total=2)
            result = store.query(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID)
            self.assertEqual(result["counts"]["total"], 2)
            self.assertEqual(result["counts"]["accounted"], 1)
            self.assertEqual(result["counts"]["unaccounted"], 1)
            self.assertIn("account for 1 of the benchmark's 2 records", result["accounting_warning"])
            self.assertEqual(result["integrity_warnings"], ["Evaluation artifact has 1 rows, but registry records 2."])

    def test_registry_and_evaluation_artifact_correct_count_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self._store_with_row(
                Path(temporary), _evaluation_row("only-row", correct=False), registry_correct=1
            )
            result = store.query(training_id=TRAINING_ID, evaluation_id=EVALUATION_ID)
            self.assertIsNone(result["accounting_warning"])
            self.assertEqual(
                result["integrity_warnings"], ["Evaluation artifact has 0 judge-correct rows, but registry records 1."]
            )


class BrowseCompPlusUrlBuilderTest(unittest.TestCase):
    def test_build_mapping_keeps_only_unique_evidence_and_positive_urls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            destination = root / "viewer" / "data" / "urls.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "query_id": 7,
                        "query": "Large question text",
                        "evidence_docs": [
                            {"url": "https://example.test/a", "text": "Large evidence"},
                            {"url": "https://example.test/a", "text": "Duplicate"},
                            {"url": "https://example.test/b", "text": "More evidence"},
                        ],
                        "gold_docs": [{"url": "https://example.test/a", "text": "Gold"}],
                        "negative_docs": [{"url": "https://example.test/negative", "text": "Negative"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            count, missing = build_mapping(source, destination)
            record = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual((count, missing), (1, 0))
            self.assertEqual(
                record,
                {
                    "query_id": "7",
                    "evidence_urls": ["https://example.test/a", "https://example.test/b"],
                    "positive_urls": ["https://example.test/a"],
                },
            )


if __name__ == "__main__":
    unittest.main()
