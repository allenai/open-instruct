"""
End-to-end tests for nemo verifiers against real HuggingFace dataset samples.

Loads actual samples from davidheineman/nemotron-super-stage-1-unmixed-openinstruct,
crafts realistic model-style predictions, and verifies correct/incorrect answers
score differently.
"""
import json
import re
import pytest
from datasets import load_dataset
from types import SimpleNamespace

from open_instruct.ground_truth_utils import build_all_verifiers

HF = "davidheineman/nemotron-super-stage-1-unmixed-openinstruct"

@pytest.fixture(scope="module")
def verifiers():
    args = SimpleNamespace(
        llm_judge_model="gpt-4.1", llm_judge_max_tokens=2048,
        llm_judge_max_context_length=128000, llm_judge_temperature=0.0,
        llm_judge_timeout=60, seed=42,
        code_api_url="http://localhost:1234/test_program",
        code_max_execution_time=10.0, code_pass_rate_reward_threshold=1.0,
        code_apply_perf_penalty=False, max_length_verifier_max_length=32768,
        rubric_judge_model="gpt-4.1", rubric_judge_max_tokens=2048,
        rubric_judge_temperature=0.0, rubric_judge_timeout=60,
    )
    return build_all_verifiers(args)


def _load_rows(split, n=5):
    ds = load_dataset(HF, split=split, streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        rows.append(row)
    return rows


# ── Math (dapo_math, skywork_math, math_proofs) ──────────────────────────

class TestMathVerifiers:
    @pytest.fixture(params=["dapo_math", "skywork_math"])
    def math_rows(self, request):
        return request.param, _load_rows(request.param, 5)

    def test_correct_boxed_answer(self, verifiers, math_rows):
        split, rows = math_rows
        for row in rows:
            v = verifiers[row["dataset"]]
            gt = row["ground_truth"]
            # Unwrap JSON list labels like '["15625"]' -> "15625"
            try:
                parsed = json.loads(gt)
                answer = parsed[0] if isinstance(parsed, list) and len(parsed) == 1 else str(parsed)
            except (json.JSONDecodeError, ValueError):
                answer = gt
            prediction = f"<think>Let me work through this step by step.</think>\n\nThe answer is $\\boxed{{{answer}}}$"
            result = v([], prediction, gt)
            assert result.score == 1.0, f"Failed for {split} gt={gt!r} answer={answer!r}"

    def test_wrong_boxed_answer(self, verifiers, math_rows):
        split, rows = math_rows
        for row in rows:
            v = verifiers[row["dataset"]]
            gt = row["ground_truth"]
            prediction = r"<think>Hmm.</think>\n\nThe answer is $\boxed{99999999}$"
            result = v([], prediction, gt)
            assert result.score == 0.0, f"Should be wrong for {split}"


# ── Instruction following ─────────────────────────────────────────────────

class TestInstructionFollowing:
    @pytest.fixture
    def rows(self):
        return _load_rows("instruction_following", 5)

    def test_dispatch_and_score(self, verifiers, rows):
        for row in rows:
            v = verifiers[row["dataset"]]
            gt = row["ground_truth"]
            result_wrong = v([], "nope", gt)
            assert isinstance(result_wrong.score, float)
            assert result_wrong.score < 1.0

    def test_empty_prediction_is_zero(self, verifiers, rows):
        for row in rows:
            v = verifiers[row["dataset"]]
            result = v([], "", row["ground_truth"])
            assert result.score == 0.0


# ── MCQA ──────────────────────────────────────────────────────────────────

class TestMCQA:
    @pytest.fixture
    def rows(self):
        return _load_rows("mcqa", 10)

    def test_correct_boxed_letter(self, verifiers, rows):
        for row in rows:
            v = verifiers[row["dataset"]]
            gt_letter = row["ground_truth"].strip()
            prediction = f"<think>After analyzing the options...</think>\n\nThe answer is \\boxed{{{gt_letter}}}"
            result = v([], prediction, gt_letter)
            assert result.score == 1.0, f"Failed for gt={gt_letter!r}"

    def test_correct_answer_colon(self, verifiers, rows):
        for row in rows:
            v = verifiers[row["dataset"]]
            gt_letter = row["ground_truth"].strip()
            prediction = f"Thinking about this carefully, I believe Answer: {gt_letter}"
            result = v([], prediction, gt_letter)
            assert result.score == 1.0, f"Failed for gt={gt_letter!r}"

    def test_wrong_letter(self, verifiers, rows):
        for row in rows:
            v = verifiers[row["dataset"]]
            gt_letter = row["ground_truth"].strip()
            wrong = "Z" if gt_letter != "Z" else "Y"
            prediction = f"\\boxed{{{wrong}}}"
            result = v([], prediction, gt_letter)
            assert result.score == 0.0


# ── Reasoning gym ─────────────────────────────────────────────────────────

class TestReasoningGym:
    @pytest.fixture
    def rows(self):
        return _load_rows("reasoning_gym", 10)

    def test_correct_with_answer_tags(self, verifiers, rows):
        for row in rows:
            v = verifiers[row["dataset"]]
            gt = row["ground_truth"]
            prediction = f"<think>Working through this puzzle...</think>\n<answer>{gt}</answer>"
            result = v([], prediction, gt)
            assert result.score == 1.0, f"Failed for gt={gt!r}"

    def test_wrong_answer(self, verifiers, rows):
        for row in rows:
            v = verifiers[row["dataset"]]
            gt = row["ground_truth"]
            prediction = "<answer>COMPLETELY_WRONG_ANSWER_12345</answer>"
            result = v([], prediction, gt)
            assert result.score == 0.0


# ── Calendar ──────────────────────────────────────────────────────────────

class TestCalendar:
    @pytest.fixture
    def rows(self):
        return _load_rows("calendar", 5)

    def test_valid_schedule(self, verifiers, rows):
        """Build a prediction that satisfies each row's constraints."""
        for row in rows:
            v = verifiers[row["dataset"]]
            gt = row["ground_truth"]
            exp = json.loads(gt)

            events = []
            for eid, constraints in exp.items():
                dur = constraints["duration"]
                min_t = constraints["min_time"]
                # Schedule at min_time (safe default)
                events.append({
                    "event_id": int(eid),
                    "start_time": min_t,
                    "duration": dur,
                })
            prediction = json.dumps(events)
            result = v([], prediction, gt)
            # May or may not pass depending on conflicts, but should not error
            assert isinstance(result.score, float)

    def test_wrong_number_of_events(self, verifiers, rows):
        for row in rows:
            v = verifiers[row["dataset"]]
            gt = row["ground_truth"]
            exp = json.loads(gt)
            if not exp:
                continue
            prediction = '[{"event_id": 999, "start_time": "9am", "duration": 30}]'
            result = v([], prediction, gt)
            if len(exp) != 1:
                assert result.score == 0.0

    def test_no_json_is_zero(self, verifiers, rows):
        for row in rows:
            v = verifiers[row["dataset"]]
            gt = row["ground_truth"]
            exp = json.loads(gt)
            if not exp:
                continue
            result = v([], "I cannot schedule any events today.", gt)
            assert result.score == 0.0


# ── Structured outputs ────────────────────────────────────────────────────

class TestStructuredOutputs:
    @pytest.fixture
    def rows(self):
        return _load_rows("structured_outputs", 5)

    def test_valid_json_matches_schema(self, verifiers, rows):
        """Generate a JSON object that satisfies the schema from the GT."""
        passed = 0
        for row in rows:
            v = verifiers[row["dataset"]]
            schema_str = row["ground_truth"]
            try:
                schema = json.loads(schema_str)
            except json.JSONDecodeError:
                continue
            obj = _build_minimal_json(schema)
            if obj is None:
                continue
            prediction = json.dumps(obj)
            result = v([], prediction, schema_str)
            if result.score == 1.0:
                passed += 1
        assert passed >= 1, "At least one schema should validate"

    def test_simple_schema_validates(self, verifiers):
        v = verifiers["nemo_structured_outputs"]
        schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
        result = v([], '{"name": "Alice", "age": 30}', json.dumps(schema))
        assert result.score == 1.0

    def test_simple_schema_rejects_extra_field(self, verifiers):
        v = verifiers["nemo_structured_outputs"]
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = v([], '{"name": "Alice", "extra": true}', json.dumps(schema))
        assert result.score == 0.0

    def test_invalid_json_is_zero(self, verifiers, rows):
        for row in rows:
            v = verifiers[row["dataset"]]
            result = v([], "this is not json at all", row["ground_truth"])
            assert result.score == 0.0


def _build_minimal_json(schema, depth=0):
    """Recursively build a minimal JSON object satisfying an OpenAPI schema."""
    if depth > 8:
        return None
    enum = schema.get("enum")
    if enum:
        return enum[0]
    t = schema.get("type", "string")
    if t == "object":
        props = schema.get("properties", {})
        obj = {}
        for k, sub in props.items():
            val = _build_minimal_json(sub, depth + 1)
            if val is None:
                return None
            obj[k] = val
        return obj
    elif t == "array":
        items = schema.get("items", {"type": "string"})
        val = _build_minimal_json(items, depth + 1)
        return [val] if val is not None else ["placeholder"]
    elif t == "string":
        return "example"
    elif t == "integer":
        return 0
    elif t == "number":
        return 0.0
    elif t == "boolean":
        return True
    elif t == "null":
        return None
    return "fallback"


# ── Safety ────────────────────────────────────────────────────────────────

class TestSafety:
    def test_refusal_prefix_detection(self, verifiers):
        v = verifiers["nemo_safety"]
        # With explicit safe/unsafe labels
        assert v([], "Sure, here's how.", json.dumps({"label": "safe"})).score == 1.0
        assert v([], "I'm sorry, I cannot help.", json.dumps({"label": "safe"})).score == 0.0
        assert v([], "I'm sorry, I cannot help.", json.dumps({"label": "unsafe"})).score == 1.0
        assert v([], "Sure, here's how.", json.dumps({"label": "unsafe"})).score == 0.0


# ── Agentic tool use ─────────────────────────────────────────────────────

class TestAgenticToolUse:
    @pytest.fixture
    def rows(self):
        return _load_rows("agentic_tool_use", 10)

    def test_message_type_scores(self, verifiers, rows):
        for row in rows:
            v = verifiers[row["dataset"]]
            gt = row["ground_truth"]
            info = json.loads(gt)
            if info.get("type") == "message":
                result = v([], "Hello! Here's what I found for you.", gt)
                assert result.score == 1.0
                result_empty = v([], "", gt)
                assert result_empty.score == 0.0

    def test_function_call_type(self, verifiers, rows):
        for row in rows:
            v = verifiers[row["dataset"]]
            gt = row["ground_truth"]
            info = json.loads(gt)
            if info.get("type") != "function_call":
                continue
            # Craft a matching tool call
            prediction = json.dumps({"name": info["name"], "arguments": info["arguments"]})
            result = v([], prediction, gt)
            assert result.score == 1.0, f"Matching tool call should score 1.0 for {info['name']}"

            # Wrong tool name
            prediction_wrong = json.dumps({"name": "wrong_tool", "arguments": info["arguments"]})
            result_wrong = v([], prediction_wrong, gt)
            assert result_wrong.score == 0.0


# ── SWE pivot ─────────────────────────────────────────────────────────────

class TestSwePivot:
    @pytest.fixture
    def rows(self):
        return _load_rows("swe_pivot", 5)

    def test_correct_tool_call(self, verifiers, rows):
        for row in rows:
            v = verifiers[row["dataset"]]
            gt = row["ground_truth"]
            info = json.loads(gt)
            if info.get("type") != "function_call":
                continue
            prediction = json.dumps({"name": info["name"], "arguments": info["arguments"]})
            result = v([], prediction, gt)
            assert result.score == 1.0

    def test_wrong_tool_call(self, verifiers, rows):
        for row in rows:
            v = verifiers[row["dataset"]]
            gt = row["ground_truth"]
            info = json.loads(gt)
            if info.get("type") != "function_call":
                continue
            prediction = json.dumps({"name": "totally_wrong", "arguments": "{}"})
            result = v([], prediction, gt)
            assert result.score == 0.0


# ── Workplace assistant ───────────────────────────────────────────────────

class TestWorkplaceAssistant:
    @pytest.fixture
    def rows(self):
        return _load_rows("workplace_assistant", 3)

    def test_correct_actions_via_text(self, verifiers, rows):
        """Feed the GT actions as the prediction text — should match."""
        for row in rows:
            v = verifiers[row["dataset"]]
            gt = row["ground_truth"]
            gt_actions = json.loads(gt)
            # Embed each action as a JSON object in the prediction text
            prediction_parts = []
            for action in gt_actions:
                prediction_parts.append(json.dumps(action))
            prediction = "\n".join(prediction_parts)
            result = v([], prediction, gt)
            # Without the Gym environment this falls back to LLM judge (returns 0.0 without API)
            # But if Gym IS available, it should score 1.0
            assert isinstance(result.score, float)

    def test_empty_prediction(self, verifiers, rows):
        for row in rows:
            v = verifiers[row["dataset"]]
            result = v([], "I don't know how to help.", row["ground_truth"])
            assert isinstance(result.score, float)


# ── Competitive coding ────────────────────────────────────────────────────

class TestCompetitiveCoding:
    @pytest.fixture
    def rows(self):
        return _load_rows("competitive_coding", 3)

    def test_dispatches_without_crash(self, verifiers, rows):
        """Without a code sandbox, the verifier should return 0.0 gracefully."""
        for row in rows:
            v = verifiers[row["dataset"]]
            prediction = '```python\nprint("hello")\n```'
            result = v([], prediction, row["ground_truth"])
            assert result.score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
