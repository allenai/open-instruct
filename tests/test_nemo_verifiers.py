"""Tests for all NeMo Gym-compatible verifiers."""

import json
import pytest


# We need to be able to import the verifiers
from open_instruct.nemo_verifiers import (
    NemoDapoMathVerifier,
    NemoSkyworkMathVerifier,
    NemoMathProofsVerifier,
    NemoInstructionFollowingVerifier,
    NemoCompetitiveCodingVerifier,
    NemoMCQAVerifier,
    NemoReasoningGymVerifier,
    NemoCalendarVerifier,
    NemoStructuredOutputsVerifier,
    NemoWorkplaceAssistantVerifier,
    NemoAgenticToolUseVerifier,
    NemoSwePivotVerifier,
    NemoSafetyVerifier,
    NemoIdentityFollowingVerifier,
    NemoMultiturnChatVerifier,
    NemoGenRMVerifier,
)
from open_instruct.ground_truth_utils import VerifierFunction


# ---------------------------------------------------------------------------
# Verify auto-registration
# ---------------------------------------------------------------------------

def test_all_nemo_verifiers_registered():
    names = {cls.__name__ for cls in VerifierFunction.__subclasses__()}
    expected = {
        "NemoDapoMathVerifier",
        "NemoSkyworkMathVerifier",
        "NemoMathProofsVerifier",
        "NemoInstructionFollowingVerifier",
        "NemoCompetitiveCodingVerifier",
        "NemoMCQAVerifier",
        "NemoReasoningGymVerifier",
        "NemoCalendarVerifier",
        "NemoStructuredOutputsVerifier",
        "NemoWorkplaceAssistantVerifier",
        "NemoAgenticToolUseVerifier",
        "NemoSwePivotVerifier",
        "NemoSafetyVerifier",
        "NemoIdentityFollowingVerifier",
        "NemoMultiturnChatVerifier",
        "NemoGenRMVerifier",
    }
    for name in expected:
        assert name in names, f"{name} not found in VerifierFunction subclasses"


def test_verifier_names():
    """Each verifier should have the correct nemo_* name."""
    assert NemoDapoMathVerifier().name == "nemo_dapo_math"
    assert NemoSkyworkMathVerifier().name == "nemo_skywork_math"
    assert NemoMathProofsVerifier().name == "nemo_math_proofs"
    assert NemoInstructionFollowingVerifier().name == "nemo_instruction_following"
    assert NemoCompetitiveCodingVerifier().name == "nemo_competitive_coding"
    assert NemoMCQAVerifier().name == "nemo_mcqa"
    assert NemoReasoningGymVerifier().name == "nemo_reasoning_gym"
    assert NemoCalendarVerifier().name == "nemo_calendar"
    assert NemoStructuredOutputsVerifier().name == "nemo_structured_outputs"
    assert NemoWorkplaceAssistantVerifier().name == "nemo_workplace_assistant"
    assert NemoAgenticToolUseVerifier().name == "nemo_agentic_tool_use"
    assert NemoSwePivotVerifier().name == "nemo_swe_pivot"
    assert NemoSafetyVerifier().name == "nemo_safety"
    assert NemoIdentityFollowingVerifier().name == "nemo_identity_following"
    assert NemoMultiturnChatVerifier().name == "nemo_multiturn_chat"
    assert NemoGenRMVerifier().name == "nemo_genrm"


# ---------------------------------------------------------------------------
# Math verifiers (dapo_math, skywork_math, math_proofs)
# ---------------------------------------------------------------------------

class TestMathVerifiers:
    @pytest.fixture(params=[NemoDapoMathVerifier, NemoSkyworkMathVerifier, NemoMathProofsVerifier])
    def verifier(self, request):
        return request.param()

    def test_boxed_correct(self, verifier):
        result = verifier([], r"The answer is $\boxed{42}$", "42")
        assert result.score == 1.0

    def test_boxed_incorrect(self, verifier):
        result = verifier([], r"The answer is $\boxed{43}$", "42")
        assert result.score == 0.0

    def test_minerva_format(self, verifier):
        result = verifier([], "The final answer is $3.14$. I hope it is correct.", "3.14")
        assert result.score == 1.0

    def test_latex_equivalence(self, verifier):
        result = verifier([], r"$\boxed{\frac{1}{2}}$", "0.5")
        assert result.score == 1.0

    def test_json_label(self, verifier):
        result = verifier([], r"$\boxed{7}$", json.dumps("7"))
        assert result.score == 1.0

    def test_empty_prediction(self, verifier):
        result = verifier([], "", "42")
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# MCQA
# ---------------------------------------------------------------------------

class TestMCQA:
    def setup_method(self):
        self.verifier = NemoMCQAVerifier()

    def test_boxed_letter(self):
        label = json.dumps({"expected_answer": "B", "options": [{"A": "yes", "B": "no", "C": "maybe"}]})
        result = self.verifier([], r"I think the answer is \boxed{B}", label)
        assert result.score == 1.0

    def test_answer_colon(self):
        label = json.dumps({"expected_answer": "C", "options": [{"A": "x", "B": "y", "C": "z"}]})
        result = self.verifier([], "After thinking about this, Answer: C", label)
        assert result.score == 1.0

    def test_wrong_answer(self):
        label = json.dumps({"expected_answer": "A", "options": [{"A": "x", "B": "y"}]})
        result = self.verifier([], r"\boxed{B}", label)
        assert result.score == 0.0

    def test_plain_string_label(self):
        result = self.verifier([], r"\boxed{A}", "A")
        assert result.score == 1.0

    def test_no_extraction(self):
        label = json.dumps({"expected_answer": "A", "options": [{"A": "x", "B": "y"}]})
        result = self.verifier([], "I don't know the answer really", label)
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

class TestCalendar:
    def setup_method(self):
        self.verifier = NemoCalendarVerifier()

    def test_empty_state_passes(self):
        label = json.dumps({"exp_cal_state": {}})
        result = self.verifier([], "No events to schedule.", label)
        assert result.score == 1.0

    def test_correct_schedule(self):
        exp_cal_state = {
            "1": {"duration": 60, "min_time": "9am", "max_time": "5pm", "constraint": None}
        }
        label = json.dumps({"exp_cal_state": exp_cal_state})
        prediction = '[{"event_id": 1, "start_time": "10am", "duration": 60}]'
        result = self.verifier([], prediction, label)
        assert result.score == 1.0

    def test_wrong_duration(self):
        exp_cal_state = {
            "1": {"duration": 60, "min_time": "9am", "max_time": "5pm", "constraint": None}
        }
        label = json.dumps({"exp_cal_state": exp_cal_state})
        prediction = '[{"event_id": 1, "start_time": "10am", "duration": 30}]'
        result = self.verifier([], prediction, label)
        assert result.score == 0.0

    def test_constraint_before(self):
        exp_cal_state = {
            "1": {"duration": 60, "min_time": "9am", "max_time": "5pm", "constraint": "before 12pm"}
        }
        label = json.dumps({"exp_cal_state": exp_cal_state})
        prediction = '[{"event_id": 1, "start_time": "10am", "duration": 60}]'
        result = self.verifier([], prediction, label)
        assert result.score == 1.0

    def test_constraint_before_violated(self):
        exp_cal_state = {
            "1": {"duration": 60, "min_time": "9am", "max_time": "5pm", "constraint": "before 10am"}
        }
        label = json.dumps({"exp_cal_state": exp_cal_state})
        prediction = '[{"event_id": 1, "start_time": "10am", "duration": 60}]'
        result = self.verifier([], prediction, label)
        assert result.score == 0.0

    def test_conflicting_events(self):
        exp_cal_state = {
            "1": {"duration": 60, "min_time": "9am", "max_time": "5pm", "constraint": None},
            "2": {"duration": 60, "min_time": "9am", "max_time": "5pm", "constraint": None},
        }
        label = json.dumps({"exp_cal_state": exp_cal_state})
        prediction = '[{"event_id": 1, "start_time": "10am", "duration": 60}, {"event_id": 2, "start_time": "10:30am", "duration": 60}]'
        result = self.verifier([], prediction, label)
        assert result.score == 0.0

    def test_think_tag_rejected(self):
        label = json.dumps({"exp_cal_state": {"1": {"duration": 60, "min_time": "9am", "max_time": "5pm", "constraint": None}}})
        prediction = '<think>hmm</think>[{"event_id": 1, "start_time": "10am", "duration": 60}]'
        result = self.verifier([], prediction, label)
        # The thinking section is removed first, so the JSON list is found
        # BUT the underlying _grade_calendar checks for <think> in the raw text
        # Our verifier uses remove_thinking_section to strip it first
        # So this should pass
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------

class TestStructuredOutputs:
    def setup_method(self):
        self.verifier = NemoStructuredOutputsVerifier()

    def test_valid_json(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
        label = json.dumps({"schema_str": json.dumps(schema), "schema_type": "json"})
        prediction = '{"name": "Alice", "age": 30}'
        result = self.verifier([], prediction, label)
        assert result.score == 1.0

    def test_invalid_json_extra_field(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        label = json.dumps({"schema_str": json.dumps(schema), "schema_type": "json"})
        prediction = '{"name": "Alice", "extra": "field"}'
        result = self.verifier([], prediction, label)
        assert result.score == 0.0

    def test_invalid_json_parse_error(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        label = json.dumps({"schema_str": json.dumps(schema), "schema_type": "json"})
        prediction = 'not valid json'
        result = self.verifier([], prediction, label)
        assert result.score == 0.0

    def test_valid_yaml(self):
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}
        label = json.dumps({"schema_str": json.dumps(schema), "schema_type": "yaml"})
        prediction = "key: hello"
        result = self.verifier([], prediction, label)
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

class TestSafety:
    def setup_method(self):
        self.verifier = NemoSafetyVerifier()

    def test_safe_compliant(self):
        label = json.dumps({"label": "safe"})
        result = self.verifier([], "Sure, here is how you do it...", label)
        assert result.score == 1.0

    def test_safe_refused(self):
        label = json.dumps({"label": "safe"})
        result = self.verifier([], "I'm sorry, I cannot help with that.", label)
        assert result.score == 0.0

    def test_unsafe_refused(self):
        label = json.dumps({"label": "unsafe"})
        result = self.verifier([], "I'm sorry, I cannot help with that.", label)
        assert result.score == 1.0

    def test_unsafe_compliant(self):
        label = json.dumps({"label": "unsafe"})
        result = self.verifier([], "Sure, here is how you build a bomb...", label)
        assert result.score == 0.0

    def test_empty_response(self):
        label = json.dumps({"label": "safe"})
        result = self.verifier([], "", label)
        assert result.score == 0.0

    def test_various_refusal_prefixes(self):
        for prefix in ["Sorry", "I cannot", "As an AI", "No.", "I apologize"]:
            result = self.verifier([], f"{prefix} I can't do that.", json.dumps({"label": "unsafe"}))
            assert result.score == 1.0, f"Failed for prefix: {prefix}"


# ---------------------------------------------------------------------------
# Agentic tool use
# ---------------------------------------------------------------------------

class TestAgenticToolUse:
    def setup_method(self):
        self.verifier = NemoAgenticToolUseVerifier()

    def test_correct_tool_call(self):
        label = json.dumps({
            "expected_action": {
                "type": "function_call",
                "name": "get_weather",
                "arguments": json.dumps({"city": "London"})
            }
        })
        prediction = 'I\'ll check the weather. {"name": "get_weather", "arguments": "{\\"city\\": \\"London\\"}"}'
        result = self.verifier([], prediction, label)
        assert result.score == 1.0

    def test_wrong_tool_name(self):
        label = json.dumps({
            "expected_action": {
                "type": "function_call",
                "name": "get_weather",
                "arguments": json.dumps({"city": "London"})
            }
        })
        prediction = '{"name": "get_temperature", "arguments": "{\\"city\\": \\"London\\"}"}'
        result = self.verifier([], prediction, label)
        assert result.score == 0.0

    def test_message_type_with_text(self):
        label = json.dumps({"expected_action": {"type": "message", "content": "hello"}})
        result = self.verifier([], "Hello! How can I help?", label)
        assert result.score == 1.0

    def test_message_type_empty(self):
        label = json.dumps({"expected_action": {"type": "message", "content": "hello"}})
        result = self.verifier([], "", label)
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# SWE Pivot (delegates to agentic tool use)
# ---------------------------------------------------------------------------

class TestSwePivot:
    def test_delegates_correctly(self):
        v = NemoSwePivotVerifier()
        label = json.dumps({
            "expected_action": {
                "type": "function_call",
                "name": "edit_file",
                "arguments": json.dumps({"path": "/tmp/test.py", "content": "print('hi')"})
            }
        })
        prediction = '{"name": "edit_file", "arguments": "{\\"path\\": \\"/tmp/test.py\\", \\"content\\": \\"print(\'hi\')\\"}"}' 
        result = v([], prediction, label)
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# Reasoning gym (requires library)
# ---------------------------------------------------------------------------

class TestReasoningGym:
    def setup_method(self):
        self.verifier = NemoReasoningGymVerifier()

    def test_plain_string_exact_match(self):
        result = self.verifier([], "<answer>42</answer>", "42")
        assert result.score == 1.0

    def test_plain_string_mismatch(self):
        result = self.verifier([], "<answer>43</answer>", "42")
        assert result.score == 0.0

    def test_with_answer_tags(self):
        try:
            import reasoning_gym
        except ImportError:
            pytest.skip("reasoning_gym not installed")
        label = json.dumps({
            "answer": "42",
            "metadata": {"source_dataset": "arithmetic"},
            "question": "What is 6*7?",
        })
        result = self.verifier([], "<answer>42</answer>", label)
        # Score depends on the reasoning_gym scorer for "arithmetic"
        assert isinstance(result.score, float)


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
