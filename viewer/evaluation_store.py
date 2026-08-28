from __future__ import annotations

import json
import re
import threading
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from viewer.training_registry import BestEvaluation, TrainingRegistry, TrainingRegistryError


class EvaluationStore:
    """Lazy reader for retained BrowseComp inference and judge artifacts."""

    _OUTCOMES = {"all", "judged_correct", "judged_incorrect", "incomplete"}
    _MATCH_LIMIT = 1_000
    _URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
    _SEARCH_RESULT_HEADING = re.compile(r"(?m)^\*\*[^\n]+\*\*\s*$")

    def __init__(self, registry: TrainingRegistry) -> None:
        self.registry = registry
        self._lock = threading.RLock()
        self._cache: dict[Path, tuple[tuple[int, int], list[dict[str, Any]], dict[str, Any], list[str]]] = {}
        self._browsecomp_plus_url_cache: tuple[tuple[int, int], dict[str, dict[str, list[str]]]] | None = None

    def refresh(self) -> None:
        with self._lock:
            self._cache.clear()
            self._browsecomp_plus_url_cache = None

    def query(
        self,
        *,
        training_id: str,
        evaluation_id: str,
        outcome: str = "all",
        search: str = "",
        sort: str = "id",
        page: int = 1,
        page_size: int = 40,
    ) -> dict[str, Any]:
        evaluation = self._evaluation(training_id, evaluation_id)
        rows, summary, integrity_warnings = self._load(evaluation)
        if outcome not in self._OUTCOMES:
            raise TrainingRegistryError(f"Unknown evaluation outcome filter: {outcome}")
        if sort not in {"id", "correctness", "tools", "steps"}:
            raise TrainingRegistryError(f"Unknown evaluation sort: {sort}")
        if page < 1 or not 1 <= page_size <= 200:
            raise TrainingRegistryError("page must be positive and page_size must be between 1 and 200")

        records = [self._summary_row(row) for row in rows]
        total = evaluation.total
        judged_correct = sum(row["outcome"] == "judged_correct" for row in records)
        judged_incorrect = sum(row["outcome"] == "judged_incorrect" for row in records)
        incomplete = sum(row["outcome"] == "incomplete" for row in records)
        accounted = judged_correct + judged_incorrect + incomplete
        unaccounted = max(total - accounted, 0)
        counts = {
            "total": total,
            "judged_correct": judged_correct,
            "judged_incorrect": judged_incorrect,
            "incomplete": incomplete,
            "accounted": accounted,
            "unaccounted": unaccounted,
        }
        accounting_warning = (
            None
            if accounted == total
            else (
                f"The three outcome categories account for {accounted} of the benchmark's {total} records"
                f" ({unaccounted} unaccounted). Inspect missing or unparseable retained judge results."
                if accounted < total
                else (
                    f"The three outcome categories account for {accounted} records, but the benchmark contains "
                    f"{total} ({accounted - total} extra categorized records). Inspect duplicate retained results."
                )
            )
        )
        if outcome != "all":
            records = [row for row in records if row["outcome"] == outcome]

        needle = search.strip().casefold()
        if needle:
            records = [
                row
                for row in records
                if needle
                in " ".join([row["query_id"], row["question"], row["reference_answer"], row["response"]]).casefold()
            ]
        if sort == "correctness":
            records.sort(key=lambda row: (row["correct"] is True, row["query_id"]))
        elif sort == "tools":
            records.sort(key=lambda row: (-row["tool_calls"], row["query_id"]))
        elif sort == "steps":
            records.sort(key=lambda row: (-row["step_count"], row["query_id"]))
        else:
            records.sort(key=lambda row: self._natural_id(row["query_id"]))

        total_filtered = len(records)
        offset = (page - 1) * page_size
        records = records[offset : offset + page_size]
        return {
            "training_id": training_id,
            "evaluation": evaluation.public(self.registry.repo_root),
            "summary": summary,
            "counts": counts,
            "accounting_warning": accounting_warning,
            "integrity_warnings": integrity_warnings,
            "filters": {"outcome": outcome, "search": search, "sort": sort},
            "records": records,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total_filtered,
                "has_more": offset + len(records) < total_filtered,
            },
        }

    def detail(
        self, *, training_id: str, evaluation_id: str, query_id: str, response_index: int | None = None
    ) -> dict[str, Any]:
        evaluation = self._evaluation(training_id, evaluation_id)
        rows, summary, integrity_warnings = self._load(evaluation)
        try:
            row = next(row for row in rows if self._query_id(row) == query_id)
        except StopIteration as error:
            raise TrainingRegistryError(f"Unknown evaluation record: {query_id}") from error
        artifact = evaluation.inference_artifact
        assert artifact is not None
        raw_path = self._raw_path(artifact.path, row)
        try:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingRegistryError(f"Could not read inference record {raw_path}: {error}") from error
        responses = raw.get("responses") or []
        if not isinstance(responses, list) or not responses:
            raise TrainingRegistryError(f"Inference record has no responses: {raw_path}")

        judged_index = self._normalize_index(artifact.judged_response_index, len(responses), "judged response")
        selected_index = (
            len(responses) - 1
            if response_index is None
            else self._normalize_index(response_index, len(responses), "response")
        )
        response = responses[selected_index]
        if not isinstance(response, dict):
            raise TrainingRegistryError(f"Response {selected_index} in {raw_path} is not a mapping")
        segments = self._segments(raw, response)
        document_relevance = self._annotate_browsecomp_plus_documents(evaluation, self._query_id(row), segments)
        reference_matches = self._literal_matches(str(row.get("correct_answer") or ""), segments)
        kind_counts = dict(Counter(segment["kind"] for segment in segments))
        judge_result = row.get("judge_result") if isinstance(row.get("judge_result"), dict) else {}
        judge_applies = selected_index == judged_index
        response_choices = [
            {
                "index": index,
                "label": f"Rollout {index + 1} of {len(responses)}",
                "judged": index == judged_index,
                "finish_reason": item.get("finish_reason") if isinstance(item, dict) else None,
                "tool_calls": sum((item.get("tool_call_counts") or {}).values()) if isinstance(item, dict) else 0,
            }
            for index, item in enumerate(responses)
        ]
        return {
            "training_id": training_id,
            "evaluation": evaluation.public(self.registry.repo_root),
            "summary": summary,
            "integrity_warnings": integrity_warnings,
            "record": self._summary_row(row),
            "raw_path": str(raw_path),
            "model": raw.get("model"),
            "selected_response_index": selected_index,
            "judged_response_index": judged_index,
            "judge_applies": judge_applies,
            "responses": response_choices,
            "response": {
                "finish_reason": response.get("finish_reason"),
                "terminal_text": response.get("response") or "",
                "tool_call_counts": response.get("tool_call_counts") or {},
                "rollout_states": response.get("rollout_states") or {},
            },
            "judge": {
                "applies_to_selected_response": judge_applies,
                "prompt": row.get("judge_prompt"),
                "output": row.get("judge_response"),
                "correct": self._summary_row(row)["correct"],
                "extracted_final_answer": judge_result.get("extracted_final_answer"),
                "reasoning": judge_result.get("reasoning"),
                "parse_error": judge_result.get("parse_error"),
                "error": judge_result.get("error"),
            },
            "segments": segments,
            "kind_counts": kind_counts,
            "reference_matches": reference_matches,
            "document_relevance": document_relevance,
        }

    def matches(
        self, *, training_id: str, evaluation_id: str, query_id: str, query: str, response_index: int | None = None
    ) -> dict[str, Any]:
        """Search the complete selected trajectory without transferring every full segment."""
        term = query.strip()
        if not term:
            raise TrainingRegistryError("Trajectory search query must not be empty")
        if len(term) > 256:
            raise TrainingRegistryError("Trajectory search query must be at most 256 characters")
        evaluation = self._evaluation(training_id, evaluation_id)
        rows, _, _ = self._load(evaluation)
        try:
            row = next(row for row in rows if self._query_id(row) == query_id)
        except StopIteration as error:
            raise TrainingRegistryError(f"Unknown evaluation record: {query_id}") from error
        artifact = evaluation.inference_artifact
        assert artifact is not None
        raw_path = self._raw_path(artifact.path, row)
        try:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingRegistryError(f"Could not read inference record {raw_path}: {error}") from error
        responses = raw.get("responses") or []
        if not isinstance(responses, list) or not responses:
            raise TrainingRegistryError(f"Inference record has no responses: {raw_path}")
        selected_index = (
            len(responses) - 1
            if response_index is None
            else self._normalize_index(response_index, len(responses), "response")
        )
        response = responses[selected_index]
        if not isinstance(response, dict):
            raise TrainingRegistryError(f"Response {selected_index} in {raw_path} is not a mapping")
        result = self._literal_matches(term, self._segments(raw, response))
        return {
            "training_id": training_id,
            "evaluation_id": evaluation_id,
            "query_id": query_id,
            "selected_response_index": selected_index,
            **result,
        }

    def segment(
        self,
        *,
        training_id: str,
        evaluation_id: str,
        query_id: str,
        segment_index: int,
        response_index: int | None = None,
    ) -> dict[str, Any]:
        """Load one complete trajectory segment without transferring every full tool result."""
        evaluation = self._evaluation(training_id, evaluation_id)
        rows, _, _ = self._load(evaluation)
        try:
            row = next(row for row in rows if self._query_id(row) == query_id)
        except StopIteration as error:
            raise TrainingRegistryError(f"Unknown evaluation record: {query_id}") from error
        artifact = evaluation.inference_artifact
        assert artifact is not None
        raw_path = self._raw_path(artifact.path, row)
        try:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingRegistryError(f"Could not read inference record {raw_path}: {error}") from error
        responses = raw.get("responses") or []
        if not isinstance(responses, list) or not responses:
            raise TrainingRegistryError(f"Inference record has no responses: {raw_path}")
        selected_index = (
            len(responses) - 1
            if response_index is None
            else self._normalize_index(response_index, len(responses), "response")
        )
        response = responses[selected_index]
        if not isinstance(response, dict):
            raise TrainingRegistryError(f"Response {selected_index} in {raw_path} is not a mapping")
        segments = self._segments(raw, response)
        if not 0 <= segment_index < len(segments):
            raise TrainingRegistryError(f"Segment index {segment_index} is outside 0..{len(segments) - 1}")
        document_relevance = self._annotate_browsecomp_plus_documents(evaluation, self._query_id(row), segments)
        segment = segments[segment_index]
        full_content = str(segment.get("_full_content") or segment.get("content") or "")
        segment["content"] = full_content
        segment["char_len"] = len(full_content)
        segment["truncated"] = False
        reference_matches = self._literal_matches(str(row.get("correct_answer") or ""), [segment])
        document_regions = []
        for region in segment.get("document_regions") or []:
            document_regions.append(
                {
                    **region,
                    "start": region.get("full_start", region["start"]),
                    "end": region.get("full_end", region["end"]),
                    "in_preview": True,
                }
            )
        segment["document_regions"] = document_regions
        return {
            "training_id": training_id,
            "evaluation_id": evaluation_id,
            "query_id": query_id,
            "selected_response_index": selected_index,
            "segment": segment,
            "reference_matches": reference_matches,
            "document_relevance": document_relevance,
        }

    def _evaluation(self, training_id: str, evaluation_id: str) -> BestEvaluation:
        training = self.registry.get(training_id)
        try:
            evaluation = next(item for item in training.evaluations if item.id == evaluation_id)
        except StopIteration as error:
            raise TrainingRegistryError(f"Unknown evaluation: {training_id}/{evaluation_id}") from error
        if evaluation.inference_artifact is None:
            raise TrainingRegistryError(f"Evaluation has no registered inference artifact: {evaluation_id}")
        return evaluation

    def _load(self, evaluation: BestEvaluation) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
        artifact = evaluation.inference_artifact
        assert artifact is not None
        results_path = artifact.path / "eval" / "evaluation_results.jsonl"
        summary_path = artifact.path / "eval" / "evaluation_summary.json"
        if not results_path.is_file() or not summary_path.is_file():
            raise TrainingRegistryError(f"Evaluation artifact is incomplete: {artifact.path}")
        signature = (results_path.stat().st_mtime_ns, results_path.stat().st_size)
        with self._lock:
            cached = self._cache.get(artifact.path)
            if cached is not None and cached[0] == signature:
                return cached[1], cached[2], cached[3]
        try:
            rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingRegistryError(f"Could not read evaluation artifact {artifact.path}: {error}") from error
        integrity_warnings = []
        if len(rows) != evaluation.total:
            integrity_warnings.append(
                f"Evaluation artifact has {len(rows)} rows, but registry records {evaluation.total}."
            )
        correct = sum(self._outcome(row) == "judged_correct" for row in rows)
        if correct != evaluation.correct:
            integrity_warnings.append(
                f"Evaluation artifact has {correct} judge-correct rows, but registry records {evaluation.correct}."
            )
        with self._lock:
            self._cache[artifact.path] = (signature, rows, summary, integrity_warnings)
        return rows, summary, integrity_warnings

    @staticmethod
    def _query_id(row: dict[str, Any]) -> str:
        return str(row.get("query_id") or row.get("id") or "")

    @classmethod
    def _summary_row(cls, row: dict[str, Any]) -> dict[str, Any]:
        outcome = cls._outcome(row)
        correct = True if outcome == "judged_correct" else False if outcome == "judged_incorrect" else None
        tool_counts = row.get("tool_call_counts") if isinstance(row.get("tool_call_counts"), dict) else {}
        response = str(row.get("response") or "")
        completed = bool(row.get("is_completed"))
        return {
            "query_id": cls._query_id(row),
            "question": str(row.get("question") or ""),
            "reference_answer": str(row.get("correct_answer") or ""),
            "response": response,
            "response_preview": " ".join(response.split())[:360],
            "correct": correct,
            "outcome": outcome,
            "completed": completed,
            "finish_reason": row.get("finish_reason"),
            "tool_call_counts": tool_counts,
            "tool_calls": sum(value for value in tool_counts.values() if isinstance(value, (int, float))),
            "step_count": int(row.get("step_count") or 0),
        }

    @classmethod
    def _outcome(cls, row: dict[str, Any]) -> str:
        if not bool(row.get("is_completed")):
            return "incomplete"
        correct = cls._correct_value(row)
        if correct is True:
            return "judged_correct"
        if correct is False:
            return "judged_incorrect"
        return "unaccounted"

    @staticmethod
    def _correct_value(row: dict[str, Any]) -> bool | None:
        judge_result = row.get("judge_result") if isinstance(row.get("judge_result"), dict) else {}
        value = judge_result.get("correct")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.casefold() in {"true", "yes"}:
                return True
            if value.casefold() in {"false", "no"}:
                return False
        return None

    @staticmethod
    def _natural_id(value: str) -> tuple[int, int | str]:
        return (0, int(value)) if value.isdigit() else (1, value.casefold())

    @staticmethod
    def _normalize_index(value: int, length: int, label: str) -> int:
        index = value + length if value < 0 else value
        if not 0 <= index < length:
            raise TrainingRegistryError(f"{label} index {value} is outside 0..{length - 1}")
        return index

    @staticmethod
    def _raw_path(root: Path, row: dict[str, Any]) -> Path:
        query_id = EvaluationStore._query_id(row)
        stale = Path(str(row.get("json_path") or row.get("input_path") or f"{query_id}.json"))
        candidates = [root / f"{query_id}.json", root / stale.name]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise TrainingRegistryError(f"Could not locate raw inference JSON for query {query_id} under {root}")

    @staticmethod
    def _segments(raw: dict[str, Any], response: dict[str, Any]) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []

        def add(kind: str, content: Any, tool_name: str | None = None, match_category: str | None = None) -> None:
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
            if not text:
                return
            char_len = len(text)
            segments.append(
                {
                    "index": len(segments),
                    "kind": kind,
                    "tool_name": tool_name,
                    "content": text[:8_000],
                    "char_len": char_len,
                    "truncated": char_len > 8_000,
                    "match_category": match_category,
                    "_full_content": text,
                }
            )

        prompt = raw.get("prompt") if isinstance(raw.get("prompt"), dict) else {}
        add("prompt", prompt.get("formatted_prompt") or prompt.get("raw_prompt") or "")
        messages = response.get("result") if isinstance(response.get("result"), list) else []
        terminal_text = str(response.get("response") or "")
        terminal_message_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], dict)
                and messages[index].get("role") == "assistant"
                and messages[index].get("content") == terminal_text
                and terminal_text
            ),
            None,
        )
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "assistant":
                add("reasoning", message.get("reasoning_content") or "", match_category="reasoning")
                function_call = message.get("function_call")
                if isinstance(function_call, dict):
                    add("tool_call", function_call.get("arguments") or "", function_call.get("name"), "tool_call")
                add(
                    "final_output" if message_index == terminal_message_index else "assistant_text",
                    message.get("content") or "",
                    match_category="final_output" if message_index == terminal_message_index else "reasoning",
                )
            elif role in {"function", "tool"}:
                add("tool_result", message.get("content") or "", message.get("name"), "tool_result")
            elif role == "user":
                add("user_text", message.get("content") or "")
        if terminal_text and terminal_message_index is None:
            add("final_output", terminal_text, match_category="final_output")
        return segments

    def _annotate_browsecomp_plus_documents(
        self, evaluation: BestEvaluation, query_id: str, segments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not evaluation.benchmark.casefold().startswith("browsecomp-plus"):
            return {"available": False, "browsecomp_plus": False}
        url_map = self._browsecomp_plus_url_map()
        record = url_map.get(query_id)
        if record is None:
            return {
                "available": False,
                "browsecomp_plus": True,
                "query_id": query_id,
                "reason": "No BrowseComp-Plus evidence URL mapping was found for this question.",
            }

        evidence_urls = record["evidence_urls"]
        positive_urls = record["positive_urls"]
        positive_normalized = {self._normalize_url(url) for url in positive_urls}
        targets: dict[str, dict[str, Any]] = {}
        for url in evidence_urls:
            normalized = self._normalize_url(url)
            if normalized:
                targets[normalized] = {"url": url, "evidence": True, "positive": normalized in positive_normalized}

        pending_calls: dict[str, deque[str]] = defaultdict(deque)
        evidence_result_count = 0
        positive_result_count = 0
        matched_evidence_urls: set[str] = set()
        matched_positive_urls: set[str] = set()
        for segment in segments:
            tool_name = str(segment.get("tool_name") or "").casefold()
            full_content = str(segment.get("_full_content") or segment.get("content") or "")
            if segment["kind"] == "tool_call":
                pending_calls[tool_name].append(full_content)
                continue
            if segment["kind"] != "tool_result":
                continue

            call_content = pending_calls[tool_name].popleft() if pending_calls[tool_name] else ""
            body_matches = self._relevant_url_occurrences(full_content, targets)
            call_matches = self._relevant_url_occurrences(call_content, targets)
            call_is_visit = tool_name == "visit" or bool(
                tool_name == "bash" and re.search(r"(?:^|[;&|]\s*)visit\s+", call_content, re.IGNORECASE)
            )
            regions = self._document_regions(full_content, body_matches, search_like=not call_is_visit)
            body_urls = {match["normalized_url"] for match in body_matches}
            if call_is_visit:
                for match in call_matches:
                    if match["normalized_url"] in body_urls:
                        continue
                    regions.append(
                        {
                            "start": 0,
                            "end": len(full_content),
                            "evidence_urls": [match["url"]],
                            "positive_urls": [match["url"]] if match["positive"] else [],
                        }
                    )
            regions = self._merge_document_regions(regions)
            if not regions:
                continue

            segment["document_regions"] = [
                {
                    **region,
                    "kind": "gold" if region["positive_urls"] else "evidence",
                    "full_start": region["start"],
                    "full_end": region["end"],
                    "in_preview": region["start"] < len(segment["content"]),
                    "start": min(region["start"], len(segment["content"])),
                    "end": min(region["end"], len(segment["content"])),
                }
                for region in regions
            ]
            segment_evidence_urls = {url for region in regions for url in region["evidence_urls"]}
            segment_positive_urls = {url for region in regions for url in region["positive_urls"]}
            segment["document_match_counts"] = {
                "evidence": len(segment_evidence_urls),
                "positive": len(segment_positive_urls),
                "evidence_only": len(segment_evidence_urls - segment_positive_urls),
            }
            evidence_result_count += 1
            positive_result_count += bool(segment_positive_urls)
            matched_evidence_urls.update(segment_evidence_urls)
            matched_positive_urls.update(segment_positive_urls)

        return {
            "available": True,
            "browsecomp_plus": True,
            "query_id": query_id,
            "evidence_url_count": len(evidence_urls),
            "positive_url_count": len(positive_urls),
            "matched_evidence_url_count": len(matched_evidence_urls),
            "matched_positive_url_count": len(matched_positive_urls),
            "evidence_result_count": evidence_result_count,
            "positive_result_count": positive_result_count,
        }

    def _browsecomp_plus_url_map(self) -> dict[str, dict[str, list[str]]]:
        path = self.registry.repo_root / "viewer" / "data" / "browsecomp_plus_urls.jsonl"
        if not path.is_file():
            return {}
        signature = (path.stat().st_mtime_ns, path.stat().st_size)
        with self._lock:
            if self._browsecomp_plus_url_cache is not None and self._browsecomp_plus_url_cache[0] == signature:
                return self._browsecomp_plus_url_cache[1]
        try:
            records = {
                str(record["query_id"]): {
                    "evidence_urls": [str(url) for url in record.get("evidence_urls") or []],
                    "positive_urls": [str(url) for url in record.get("positive_urls") or []],
                }
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
                for record in [json.loads(line)]
            }
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise TrainingRegistryError(f"Could not read BrowseComp-Plus URL map {path}: {error}") from error
        with self._lock:
            self._browsecomp_plus_url_cache = (signature, records)
        return records

    @staticmethod
    def _normalize_url(value: str) -> str:
        candidate = value.strip().rstrip(".,;:")
        while candidate.endswith((")", "]", "}")):
            opening = {")": "(", "]": "[", "}": "{"}[candidate[-1]]
            if candidate.count(opening) >= candidate.count(candidate[-1]):
                break
            candidate = candidate[:-1]
        try:
            parsed = urlsplit(candidate)
        except ValueError:
            return ""
        if not parsed.netloc:
            return ""
        host = parsed.netloc.casefold()
        if host.startswith("www."):
            host = host[4:]
        path = unquote(parsed.path).rstrip("/") or "/"
        return f"{host}{path}{'?' + parsed.query if parsed.query else ''}"

    @classmethod
    def _relevant_url_occurrences(cls, content: str, targets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        matches = []
        for occurrence in cls._URL_PATTERN.finditer(content):
            raw_url = occurrence.group(0)
            normalized_url = cls._normalize_url(raw_url)
            target = targets.get(normalized_url)
            if target is None:
                continue
            trimmed_url = raw_url
            while trimmed_url and cls._normalize_url(trimmed_url) == normalized_url:
                shorter = trimmed_url[:-1]
                if not shorter or cls._normalize_url(shorter) != normalized_url:
                    break
                trimmed_url = shorter
            matches.append(
                {
                    "start": occurrence.start(),
                    "end": occurrence.start() + len(trimmed_url),
                    "normalized_url": normalized_url,
                    "url": target["url"],
                    "positive": target["positive"],
                }
            )
        return matches

    @classmethod
    def _document_regions(
        cls, content: str, matches: list[dict[str, Any]], *, search_like: bool
    ) -> list[dict[str, Any]]:
        headings = list(cls._SEARCH_RESULT_HEADING.finditer(content)) if search_like else []
        regions = []
        for match in matches:
            start = 0
            end = len(content)
            if headings:
                previous = [heading for heading in headings if heading.start() <= match["start"]]
                if previous:
                    heading_index = headings.index(previous[-1])
                    start = headings[heading_index].start()
                    if heading_index + 1 < len(headings):
                        end = headings[heading_index + 1].start()
            regions.append(
                {
                    "start": start,
                    "end": end,
                    "evidence_urls": [match["url"]],
                    "positive_urls": [match["url"]] if match["positive"] else [],
                }
            )
        return regions

    @staticmethod
    def _merge_document_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[tuple[int, int], dict[str, Any]] = {}
        for region in regions:
            key = (region["start"], region["end"])
            target = merged.setdefault(
                key, {"start": region["start"], "end": region["end"], "evidence_urls": [], "positive_urls": []}
            )
            for field in ("evidence_urls", "positive_urls"):
                target[field] = list(dict.fromkeys([*target[field], *region[field]]))
        return sorted(merged.values(), key=lambda region: (region["start"], region["end"]))

    @classmethod
    def _literal_matches(cls, term: str, segments: list[dict[str, Any]]) -> dict[str, Any]:
        counts = {category: 0 for category in ("reasoning", "tool_call", "tool_result", "final_output")}
        matches: list[dict[str, Any]] = []
        pattern = re.compile(re.escape(term), re.IGNORECASE) if term else None
        for segment in segments:
            content = str(segment.pop("_full_content", segment.get("content") or ""))
            category = segment.get("match_category")
            segment_match_count = 0
            segment_matches = () if category not in counts or pattern is None else pattern.finditer(content)
            for match in segment_matches:
                segment_match_count += 1
                if len(matches) >= cls._MATCH_LIMIT:
                    continue
                excerpt_start = max(0, match.start() - 100)
                excerpt_end = min(len(content), match.end() + 100)
                matches.append(
                    {
                        "segment_index": segment["index"],
                        "segment_kind": segment["kind"],
                        "category": category,
                        "start": match.start(),
                        "end": match.end(),
                        "excerpt": content[excerpt_start:excerpt_end],
                        "excerpt_match_start": match.start() - excerpt_start,
                        "excerpt_match_end": match.end() - excerpt_start,
                        "in_preview": match.end() <= len(segment["content"]),
                    }
                )
            segment["reference_answer_match_count"] = segment_match_count
            if category in counts:
                counts[category] += segment_match_count
        total = sum(counts.values())
        return {
            "term": term,
            "total": total,
            "counts": counts,
            "matches": matches,
            "returned": len(matches),
            "truncated": total > len(matches),
            "limit": cls._MATCH_LIMIT,
        }
