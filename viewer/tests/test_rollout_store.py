from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from viewer.rollout_store import (
    RolloutStore,
    RolloutStoreError,
    annotate_groups,
    answer_declarations,
    assess_gibberish,
    extract_json_value,
    incomplete_reason,
    json_array_length,
    question_from_prompt,
    segment_trajectory,
    terminal_prose,
)

# Mirrors a stored response: the prompt already opened <think>, so the first
# reasoning block closes without ever opening, and tool results come back as
# user turns wrapping <tool_response>.
TRAJECTORY = (
    "Let me plan the search.\n"
    "</think>\n"
    "<tool_call>\n<function=search>\n<parameter=query>\nfirst query\n</parameter>\n</function>\n</tool_call>\n"
    "<tool_call>\n<function=visit>\n<parameter=url>\nhttps://example.test\n</parameter>\n</function>\n</tool_call>"
    "<|im_end|><|im_start|>user\n<tool_response>\nfirst observation\n</tool_response>\n"
    "<tool_response>\nsecond observation\n</tool_response>\n<|im_end|>\n"
    "<|im_start|>assistant\n<think>\nNow I know the answer.\n</think>\n\n"
    "Answer: **Alpha**<|im_end|>"
)



try:  # The terminal-RL branch ships no-op shims for the browsecomp format gates.
    from open_instruct.ground_truth_utils import apply_browsecomp_format_gates  # noqa: F401

    HAS_BROWSECOMP_GATES = True
except ImportError:
    HAS_BROWSECOMP_GATES = False

requires_browsecomp_gates = unittest.skipUnless(
    HAS_BROWSECOMP_GATES, "browsecomp format gates are shimmed on this branch"
)


class GibberishAssessmentTest(unittest.TestCase):
    def test_coherent_foreign_title_is_not_gibberish(self) -> None:
        assessment = assess_gibberish("The film is 鼠膽龍威, also called High Risk; thereʼs no ambiguity.")
        self.assertEqual(assessment.reasons, [])

    def test_one_mixed_script_word_is_localized_corruption(self) -> None:
        assessment = assess_gibberish("The evidence points to The经理招聘 as the title.")
        self.assertEqual(assessment.tiers, ["localized_corruption"])
        self.assertIn("The经理招聘", assessment.localized_corruption[0])

    def test_multiple_mixed_script_words_are_token_salad(self) -> None:
        assessment = assess_gibberish("The经理招聘 was powered航空 according to the response.")
        self.assertEqual(assessment.tiers, ["token_salad"])
        self.assertIn("multiple mixed-script tokens", assessment.token_salad[0])

    def test_scattered_short_fragments_from_three_scripts_are_token_salad(self) -> None:
        latin_context = "This is otherwise ordinary English research prose with supporting evidence. " * 5
        assessment = assess_gibberish(f"{latin_context} 中 middle ж middle α")
        self.assertEqual(assessment.tiers, ["token_salad"])
        self.assertIn("scattered non-Latin script fragments", assessment.token_salad)

    def test_existing_hard_corruption_and_repetition_are_tiered(self) -> None:
        hard = assess_gibberish("reason Ġ Ġ Ġ token")
        repeated = assess_gibberish("try again try again try again try again")
        self.assertEqual(hard.tiers, ["hard_corruption"])
        self.assertEqual(repeated.tiers, ["token_salad"])


def record(
    step: int,
    sample: int,
    *,
    reward: float,
    terminal: str,
    ground_truth: str = "Alpha",
    finish_reason: str = "stop",
    calls: int = 2,
    timeouts: int = 0,
    tool_errors: str = "",
    response_size: int = 8,
    filtered: bool = False,
    termination_reason: str = "generation_complete",
    generation_finish_reason: str = "stop",
    successful_calls: int | None = None,
    verifier_input: str | None = None,
    verifier_skipped_reason: str | None = None,
    judge_output: dict | None = None,
) -> dict:
    succeeded = calls if successful_calls is None else successful_calls
    value = {
        "step": step,
        "sample_idx": sample,
        "prompt_idx": sample // 2,
        "prompt_tokens": [1, 2, 3],
        "response_tokens": list(range(response_size)),
        "reward": reward,
        "advantage": reward - 0.5,
        "finish_reason": finish_reason,
        "dataset": ["re_search"],
        "ground_truth": [ground_truth],
        "request_info": {
            "num_calls": calls,
            "timeouts": timeouts,
            "tool_errors": tool_errors,
            "tool_outputs": "Source: https://example.test",
            "tool_call_stats": [
                {"tool_name": "search", "success": index < succeeded, "runtime": 0.1} for index in range(calls)
            ],
            "rollout_state": {
                "terminal_model_text": terminal,
                "termination_reason": termination_reason,
                "generation_finish_reason": generation_finish_reason,
            },
        },
        "logprobs": [-0.1] * response_size,
    }
    rollout_state = value["request_info"]["rollout_state"]
    if verifier_input is not None:
        rollout_state["verifier_input"] = verifier_input
    if verifier_skipped_reason is not None:
        rollout_state["verifier_skipped_reason"] = verifier_skipped_reason
    if judge_output is not None:
        rollout_state["judge_output"] = judge_output
    if filtered:
        value = {
            "step": step,
            "filter_reason": "zero_std_reward",
            "sample_idx": sample,
            "prompt_idx": 0,
            "prompt_id": "1_42",
            "dataset_index": 42,
            "model_step": step - 1,
            "prompt_tokens": [1, 2, 3],
            "raw_prompt": "user: Which answer is correct?",
            "response_tokens": list(range(response_size)),
            "decoded_response": f"decoded: {terminal}",
            "reward": reward,
            "finish_reason": finish_reason,
            "dataset": ["re_search"],
            "ground_truth": [ground_truth],
            "active_tools": ["search", "visit"],
            "request_info": value["request_info"],
            "logprobs": [-0.1] * response_size,
            "reward_metrics": {"correct": reward},
        }
    return value


def terminal_record(
    step: int,
    sample: int,
    *,
    reward: float,
    finish_reason: str,
    response_size: int = 8,
    calls: int = 3,
    tool_output: str = "(exit_code=0)ok",
    tool_error: str = "",
    done: bool = True,
) -> dict:
    """A swerl/terminal env shard row: dataset=passthrough, task-id ground truth, no terminal text."""
    return {
        "step": step,
        "sample_idx": sample,
        "prompt_idx": sample // 2,
        "prompt_tokens": [1, 2, 3],
        "response_tokens": list(range(response_size)),
        "reward": reward,
        "advantage": reward - 0.5,
        "finish_reason": finish_reason,
        "dataset": ["passthrough"],
        "ground_truth": ["task_000123_abcd"],
        "request_info": {
            "num_calls": calls,
            "timeouts": False,
            "tool_errors": "",
            "tool_call_stats": [{"tool_name": "bash", "success": True, "runtime": 0.1} for _ in range(calls)],
            "rollout_state": {
                "rewards": [0.0, reward],
                "step_count": 2,
                "done": done,
                "tool_output": tool_output,
                "tool_error": tool_error,
            },
        },
        "logprobs": [-0.1] * response_size,
    }


class TerminalPolicyTest(unittest.TestCase):
    """Env-verified terminal runs classify by reward + truncation, not by answer text."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = "swerl_experiment__42__123"
        rows = [
            terminal_record(0, 0, reward=1.0, finish_reason="stop"),
            terminal_record(0, 1, reward=0.0, finish_reason="stop"),
            terminal_record(0, 2, reward=0.0, finish_reason="length", response_size=64),
            terminal_record(
                0,
                3,
                reward=0.0,
                finish_reason="stop",
                calls=1,
                done=False,
                tool_error="podman reset failed",
                tool_output=(
                    "(exit_code=0)Command timed out after 120 seconds"
                    + "(exit_code=1)same failing output" * 3
                    + "(exit_code=1)other"
                ),
            ),
        ]
        with (self.root / f"{self.run}_rollouts_000000.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        self.store = RolloutStore(self.root, response_limit=64)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def query_records(self) -> list[dict]:
        return self.store.query(run=self.run, step=0, page_size=10, category="all")["records"]

    def test_reward_and_truncation_drive_outcomes(self) -> None:
        by_sample = {record["sample_idx"]: record for record in self.query_records()}
        self.assertEqual(by_sample[0]["outcome"], "judged_correct")
        self.assertEqual(by_sample[1]["outcome"], "judged_incorrect")
        self.assertIn("stopped_wrong", by_sample[1]["categories"])
        self.assertEqual(by_sample[2]["outcome"], "incomplete")
        self.assertEqual(by_sample[2]["incomplete_reason"], "token_budget")
        self.assertIn("truncated", by_sample[2]["categories"])
        for record in by_sample.values():
            self.assertEqual(record["verifier_policy"], "terminal")

    def test_text_screens_do_not_fire_without_terminal_text(self) -> None:
        for record in self.query_records():
            self.assertNotIn("no_final_answer", record["categories"])
            self.assertNotIn("judge_positive_no_answer", record["categories"])
            self.assertNotIn("judge_negative_has_answer", record["categories"])
            self.assertNotIn("gibberish", record["categories"])

    def test_env_steps_are_exposed(self) -> None:
        self.assertEqual({record["env_steps"] for record in self.query_records()}, {2})

    def test_terminal_failure_patterns_are_flagged(self) -> None:
        by_sample = {record["sample_idx"]: record for record in self.query_records()}
        sick = by_sample[3]
        self.assertIn("bash_timeout", sick["categories"])
        self.assertEqual(sick["bash_timeouts"], 1)
        self.assertIn("mostly_failing_commands", sick["categories"])
        self.assertIn("env_not_done", sick["categories"])
        self.assertIn("gave_up_early", sick["categories"])
        self.assertIn("repetitive_commands", sick["categories"])
        self.assertIn("reset_failure", sick["categories"])
        healthy = by_sample[0]
        for category in (
            "bash_timeout",
            "mostly_failing_commands",
            "env_not_done",
            "gave_up_early",
            "repetitive_commands",
            "reset_failure",
        ):
            self.assertNotIn(category, healthy["categories"])


class RolloutStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = "experiment__42__123"
        accepted = [
            record(
                0,
                0,
                reward=1,
                terminal="Checked the evidence.</think><answer>Alpha</answer><|im_end|>",
                response_size=70_000,
                verifier_input="Judge prompt with response Alpha",
                judge_output={"score": 1.0, "attempts": [{"content": "true", "verdict": 1.0}]},
            ),
            record(0, 1, reward=0, terminal="I found it.</think><answer>Alpha</answer><|im_end|>"),
            record(0, 2, reward=1, terminal="I found it.</think><answer>Beta</answer><|im_end|>"),
            record(
                1, 0, reward=0, terminal="word word word word word word word</think><answer>Omega</answer><|im_end|>"
            ),
            record(
                1, 1, reward=1, terminal="Checked it.</think><answer>Delta</answer><|im_end|>", ground_truth="Delta"
            ),
            # Ran out of response budget, so the judge never scored it.
            record(
                3,
                0,
                reward=0,
                terminal="Finished at the boundary.</think><answer>Alpha</answer><answer>Beta</answer><|im_end|>",
                finish_reason="length",
                termination_reason="response_limit",
                response_size=80_000,
                verifier_skipped_reason="termination_reason:response_limit",
                judge_output={"skipped": "termination_reason:response_limit", "score": 0.0},
            ),
            # Failed calls are still calls; only a literal zero belongs in No Tool Calls.
            record(
                4, 0, reward=0, terminal="Checked.</think><answer>Beta</answer><|im_end|>", calls=3, successful_calls=0
            ),
            record(5, 0, reward=0, terminal="Answered from memory.</think><answer>Beta</answer><|im_end|>", calls=0),
            record(
                6, 0, reward=0, terminal="reason</think><answer>Alpha</answer><answer>Beta</answer>", response_size=7
            ),
            record(7, 0, reward=0, terminal="reason Ġ Ġ Ġ token</think><answer>Beta</answer><|im_end|>", timeouts=2),
            record(8, 0, reward=0, terminal=""),
        ]
        accepted_path = self.root / f"{self.run}_rollouts_000000.jsonl"
        with accepted_path.open("w", encoding="utf-8") as handle:
            for row in accepted:
                handle.write(json.dumps(row) + "\n")
        filtered_dir = self.root / "filtered"
        filtered_dir.mkdir()
        filtered_path = filtered_dir / f"{self.run}_filtered_rollouts_000000.jsonl"
        with filtered_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(record(2, 0, reward=0, terminal="No answer", filtered=True)) + "\n")
        metadata = {
            "run_name": self.run,
            "git_commit": "abc123",
            "model_name": "local-test-tokenizer",
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
        (self.root / f"{self.run}_metadata.jsonl").write_text(json.dumps(metadata) + "\n")
        self.store = RolloutStore(self.root, response_limit=8)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_step_ranges_stay_unresolved_until_a_run_is_opened(self) -> None:
        meta = self.store.meta()
        self.assertEqual(meta["default_run"], self.run)
        self.assertFalse(meta["runs"][0]["resolved"])
        self.assertIsNone(meta["runs"][0]["first_step"])
        self.assertIsNone(meta["runs"][0]["last_step"])
        self.store.steps(self.run)
        resolved = self.store.meta()["runs"][0]
        self.assertTrue(resolved["resolved"])
        self.assertEqual((resolved["first_step"], resolved["last_step"]), (0, 8))

    def test_meta_and_steps_cover_accepted_and_filtered_ranges(self) -> None:
        steps = self.store.steps(self.run)
        self.assertEqual(steps["steps"], list(range(9)))
        self.assertEqual(steps["source_ranges"]["accepted"], {"first_step": 0, "last_step": 8})
        self.assertEqual(steps["source_ranges"]["filtered"], {"first_step": 2, "last_step": 2})

    @requires_browsecomp_gates
    def test_query_classifies_and_paginates_without_returning_arrays(self) -> None:
        result = self.store.query(run=self.run, step=0, category="review", page_size=1)
        self.assertEqual(result["stats"]["records"], 3)
        self.assertEqual(result["total"], 2)
        self.assertTrue(result["has_more"])
        self.assertNotIn("response_tokens", result["records"][0])
        self.assertIn("judge_positive_no_answer", result["category_counts"])
        self.assertIn("judge_negative_has_answer", result["category_counts"])

    @requires_browsecomp_gates
    def test_literal_ground_truth_in_zero_reward_is_judge_review_candidate(self) -> None:
        result = self.store.query(run=self.run, step=0, category="judge_negative_has_answer")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["records"][0]["sample_idx"], 1)

    def test_binary_step_seek_converges_with_lines_larger_than_search_window(self) -> None:
        result = self.store.query(run=self.run, step=1, category="all")
        self.assertEqual(result["total"], 2)

    @requires_browsecomp_gates
    def test_rewarded_without_exact_reference_is_judge_review_candidate(self) -> None:
        result = self.store.query(run=self.run, step=0, category="judge_positive_no_answer")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["records"][0]["sample_idx"], 2)

    def test_detail_reads_one_record_by_offset(self) -> None:
        result = self.store.query(run=self.run, step=0, category="all")
        record_summary = next(record for record in result["records"] if record["sample_idx"] == 0)
        detail = self.store.detail(record_summary["id"])
        self.assertEqual(detail["step"], 0)
        self.assertIn("terminal_response", detail)
        self.assertEqual(detail["verifier_input"], "Judge prompt with response Alpha")
        self.assertEqual(detail["judge_output"]["score"], 1.0)
        self.assertIsNone(detail["verifier_skipped_reason"])
        self.assertNotIn("response_tokens", detail)

    def test_detail_exposes_verifier_skip_reason(self) -> None:
        result = self.store.query(run=self.run, step=3, category="all")
        detail = self.store.detail(result["records"][0]["id"])

        self.assertEqual(detail["verifier_skipped_reason"], "termination_reason:response_limit")
        self.assertIsNone(detail["verifier_input"])
        self.assertEqual(detail["judge_output"]["skipped"], "termination_reason:response_limit")

    def test_filtered_trace_uses_saved_decoded_response(self) -> None:
        result = self.store.query(run=self.run, step=2, source="filtered", category="all")
        self.assertEqual(result["total"], 1)
        trace = self.store.trace(result["records"][0]["id"], limit=1_000)
        self.assertIn("decoded: No answer", trace["content"])
        self.assertFalse(trace["has_more"])

    def test_targeted_extractors_do_not_materialize_unrelated_fields(self) -> None:
        line = json.dumps(record(0, 0, reward=1, terminal="Answer: Alpha")).encode()
        self.assertEqual(extract_json_value(line, "reward"), 1)
        self.assertEqual(extract_json_value(line, "terminal_model_text"), "Answer: Alpha")
        self.assertEqual(json_array_length(line, "response_tokens"), 8)

    def test_markdown_bold_answer_heading_extracts_only_the_value(self) -> None:
        self.assertEqual(answer_declarations("**Final Answer:** The Long Goodbye"), ["The Long Goodbye"])

    def test_zero_reward_splits_into_judged_and_never_judged(self) -> None:
        judged = self.store.query(run=self.run, step=0, category="judged_incorrect")
        self.assertEqual([r["sample_idx"] for r in judged["records"]], [1])
        capped = self.store.query(run=self.run, step=3, category="incomplete")
        self.assertEqual(capped["total"], 1)
        self.assertEqual(capped["records"][0]["incomplete_reason"], "response_limit")
        self.assertIn("incomplete_response_limit", capped["records"][0]["categories"])
        self.assertEqual(self.store.query(run=self.run, step=3, category="judged_incorrect")["total"], 0)

    def test_group_shape_is_available_as_a_queue(self) -> None:
        # Step 0 holds one all-correct pair and one mixed pair.
        result = self.store.query(run=self.run, step=0, category="all_correct_group")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["records"][0]["group_correct"], 1)
        mixed = self.store.query(run=self.run, step=0, category="mixed_group")
        self.assertEqual(mixed["total"], 2)

    def test_group_view_summarizes_pass_rates_and_difficulty(self) -> None:
        result = self.store.groups(run=self.run, step=0)

        self.assertEqual(result["stats"]["groups"], 2)
        self.assertEqual(result["stats"]["mean_group_pass_rate"], 0.75)
        self.assertEqual([group["difficulty"] for group in result["groups"]], ["learning_group", "all_correct_group"])
        self.assertEqual(result["groups"][0]["correct"], 1)
        self.assertEqual(result["groups"][0]["size"], 2)

    def test_group_view_can_filter_and_trajectory_query_can_focus_one_group(self) -> None:
        result = self.store.groups(run=self.run, step=0, category="learning_group")
        self.assertEqual(result["total"], 1)
        group = result["groups"][0]

        trajectories = self.store.query(run=self.run, step=0, category="all", group_key=group["group_key"])
        self.assertEqual(trajectories["total"], 2)
        self.assertTrue(all(record["group_key"] == group["group_key"] for record in trajectories["records"]))

    def test_ordinary_zero_reward_and_length_no_longer_look_suspicious(self) -> None:
        healthy = self.store.query(run=self.run, step=3, category="healthy_looking")
        self.assertEqual(healthy["total"], 1)
        self.assertEqual(healthy["records"][0]["suspicion_score"], 0)
        self.assertIn("token_capped", healthy["records"][0]["categories"])
        self.assertIn("long", healthy["records"][0]["categories"])

    def test_no_tool_calls_means_zero_calls_not_zero_successes(self) -> None:
        self.assertEqual(self.store.query(run=self.run, step=4, category="no_tool_calls")["total"], 0)
        result = self.store.query(run=self.run, step=5, category="no_tool_calls")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["records"][0]["successful_tool_calls"], 0)

    @requires_browsecomp_gates
    def test_format_gibberish_and_timeout_flags_are_separate(self) -> None:
        malformed = self.store.query(run=self.run, step=6, category="format_error")
        self.assertEqual(malformed["total"], 1)
        self.assertEqual(malformed["records"][0]["format_error_reason"], "answer_open_count:2")
        gibberish = self.store.query(run=self.run, step=7, category="gibberish")
        self.assertEqual(gibberish["total"], 1)
        self.assertIn("raw tokenizer artifacts", "\n".join(gibberish["records"][0]["reasons"]))
        self.assertEqual(self.store.query(run=self.run, step=7, category="timeouts")["total"], 1)

    @requires_browsecomp_gates
    def test_token_capped_trajectory_is_excluded_from_format_error_queue(self) -> None:
        capped = self.store.query(run=self.run, step=3, category="all")["records"][0]
        self.assertIsNotNone(capped["format_error_reason"])
        self.assertIn("token_capped", capped["categories"])
        self.assertNotIn("format_error", capped["categories"])
        self.assertEqual(self.store.query(run=self.run, step=3, category="format_error")["total"], 0)

    def test_terminal_prose_preserves_all_reasoning_and_irregular_markers(self) -> None:
        raw = "first</think>answer<think>second</think>final<tool_call>bad</tool_call><|im_end|>"
        self.assertEqual(terminal_prose(raw), raw)

    def test_no_final_answer_means_no_terminal_turn(self) -> None:
        self.assertEqual(self.store.query(run=self.run, step=8, category="no_final_answer")["total"], 1)
        self.assertEqual(self.store.query(run=self.run, step=3, category="no_final_answer")["total"], 0)

    def test_turns_prefixes_the_prompt_and_truncates_long_segments(self) -> None:
        result = self.store.query(run=self.run, step=2, source="filtered", category="all")
        turns = self.store.turns(result["records"][0]["id"], max_chars_per_turn=500)
        self.assertEqual(turns["segments"][0]["kind"], "prompt")
        self.assertEqual(turns["segments"][0]["content"], "user: Which answer is correct?")
        self.assertTrue(all(len(segment["content"]) <= 500 for segment in turns["segments"]))
        self.assertEqual([segment["index"] for segment in turns["segments"]], list(range(turns["total_segments"])))

    def test_turns_index_reference_answer_and_mark_final_assistant_text(self) -> None:
        result = self.store.query(run=self.run, step=2, source="filtered", category="all")
        record_id = result["records"][0]["id"]
        _, file_info, line = self.store._record_line(record_id)
        row = json.loads(line)
        row["ground_truth"] = ["No answer"]
        row["decoded_response"] = "considering No answer</think>Final Answer: No answer"
        file_info.path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.store.refresh()
        record_id = self.store.query(run=self.run, step=2, source="filtered", category="all")["records"][0]["id"]

        turns = self.store.turns(record_id, max_chars_per_turn=500)

        self.assertEqual(turns["segments"][-1]["kind"], "final_output")
        self.assertEqual(turns["reference_matches"]["counts"]["reasoning"], 1)
        self.assertEqual(turns["reference_matches"]["counts"]["final_output"], 1)
        self.assertEqual(turns["reference_matches"]["total"], 2)

    def test_trajectory_search_is_case_insensitive_and_finds_text_beyond_preview(self) -> None:
        result = self.store.query(run=self.run, step=2, source="filtered", category="all")
        record_id = result["records"][0]["id"]
        _, file_info, line = self.store._record_line(record_id)
        row = json.loads(line)
        row["decoded_response"] = f"{'x' * 700} Hidden Needle"
        file_info.path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.store.refresh()
        record_id = self.store.query(run=self.run, step=2, source="filtered", category="all")["records"][0]["id"]

        matches = self.store.matches(record_id, "hidden needle", max_chars_per_turn=500)

        self.assertEqual(matches["total"], 1)
        self.assertEqual(matches["counts"]["final_output"], 1)
        self.assertFalse(matches["matches"][0]["in_preview"])

    def test_trajectory_search_rejects_empty_and_oversized_queries(self) -> None:
        result = self.store.query(run=self.run, step=2, source="filtered", category="all")
        record_id = result["records"][0]["id"]
        with self.assertRaisesRegex(RolloutStoreError, "must not be empty"):
            self.store.matches(record_id, "  ")
        with self.assertRaisesRegex(RolloutStoreError, "at most 256"):
            self.store.matches(record_id, "x" * 257)


class RestartLineageTest(unittest.TestCase):
    class EvalIndex:
        def lineage(self, run_name, directory):
            return {"id": "wandb123", "name": "experiment__42__200"}

        def evaluations(self, run_name, directory, run_id=None):
            return [
                {
                    "artifact_step": 0,
                    "optimizer_step": 1,
                    "score": 0.25,
                    "metric": "eval/objective/verifiable_correct_rate",
                    "wandb_run_id": "wandb123",
                },
                {
                    "artifact_step": 2,
                    "optimizer_step": 3,
                    "score": 0.5,
                    "metric": "eval/objective/verifiable_correct_rate",
                    "wandb_run_id": "wandb123",
                },
            ]

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "pool_a"
        self.root.mkdir()
        attempts = {
            "experiment__42__100": [
                record(0, 0, reward=0, terminal="Old zero"),
                record(1, 0, reward=0, terminal="Old one"),
                record(2, 0, reward=0, terminal="Old overlap"),
            ],
            "experiment__42__200": [
                record(2, 10, reward=1, terminal="New overlap"),
                record(3, 10, reward=1, terminal="New three"),
            ],
        }
        for attempt, rows in attempts.items():
            with (self.root / f"{attempt}_rollouts_000000.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            metadata = {
                "run_name": attempt,
                "git_commit": "abc123",
                "model_name": "local-test-tokenizer",
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
            (self.root / f"{attempt}_metadata.jsonl").write_text(json.dumps(metadata) + "\n")
        self.store = RolloutStore(self.root, eval_index=self.EvalIndex())

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_attempts_are_consolidated_and_newer_overlap_wins(self) -> None:
        meta = self.store.meta()
        self.assertEqual(len(meta["runs"]), 1)
        logical_run = meta["runs"][0]["name"]
        self.assertEqual(logical_run, "wandb:wandb123")
        self.assertEqual(meta["runs"][0]["attempts"], ["experiment__42__100", "experiment__42__200"])
        self.assertEqual(self.store.steps(logical_run)["steps"], [0, 1, 2, 3])
        overlap = self.store.query(run=logical_run, step=2, category="all")
        self.assertEqual(overlap["stats"]["records"], 1)
        self.assertEqual(overlap["records"][0]["sample_idx"], 10)

    def test_validation_scores_span_the_consolidated_lineage(self) -> None:
        logical_run = self.store.meta()["runs"][0]["name"]
        steps = self.store.steps(logical_run)
        self.assertEqual(steps["evaluated_steps"], [0, 2])
        self.assertEqual([item["score"] for item in steps["evaluations"]], [0.25, 0.5])


class HistoricalVerifierInputTest(unittest.TestCase):
    class EvalIndex:
        def lineage(self, run_name, directory):
            return {
                "id": "format-gates",
                "name": "Format-gated LLM judge",
                "tags": {"verifier": "LLM judge", "judge": "single-pass format gates"},
            }

    class Tokenizer:
        def decode(self, tokens, skip_special_tokens=False):
            del tokens, skip_special_tokens
            return (
                "<|im_start|>system\nTools live here<|im_end|>\n"
                "<|im_start|>user\nWhich answer is correct?<|im_end|>\n"
                "<|im_start|>assistant\n<think>"
            )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = "format_gates__42__123"
        rows = [
            record(0, 0, reward=1, terminal="private reasoning</think>Alpha<|im_end|>"),
            record(1, 0, reward=0, terminal="Alpha"),
            record(
                2,
                0,
                reward=1,
                terminal="private reasoning</think>Alpha<|im_end|>",
                verifier_input="authoritative saved prompt",
            ),
            record(
                3,
                0,
                reward=0,
                terminal="private reasoning</think>Alpha<|im_end|>",
                termination_reason="response_limit",
                generation_finish_reason="length",
            ),
            record(4, 0, reward=0, terminal="Alpha appears only in reasoning</think>Beta<|im_end|>"),
            record(5, 0, reward=0, terminal="Checked it</think>Alpha<|im_end|>"),
            record(6, 0, reward=1, terminal="Checked it</think>Beta<|im_end|>"),
            record(7, 0, reward=0, terminal="first</think>second</think>Alpha<|im_end|>"),
        ]
        self.path = self.root / f"{self.run}_rollouts_000000.jsonl"
        self.path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))
        self.store = RolloutStore(self.root, tokenizer_name="local-test-tokenizer", eval_index=self.EvalIndex())
        self.store._tokenizer = self.Tokenizer()
        self.logical_run = self.store.meta()["runs"][0]["name"]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def detail(self, step: int) -> dict:
        result = self.store.query(run=self.logical_run, step=step, category="all")
        return self.store.detail(result["records"][0]["id"])

    @requires_browsecomp_gates
    def test_missing_input_is_reconstructed_with_current_format_gate_logic(self) -> None:
        before = self.path.read_bytes()
        detail = self.detail(0)

        self.assertEqual(detail["verifier_input_source"], "reconstructed")
        self.assertIn("[question]: Which answer is correct?", detail["verifier_input"])
        self.assertIn("[reference_answer]: Alpha", detail["verifier_input"])
        self.assertIn("[response]: Alpha", detail["verifier_input"])
        self.assertNotIn("private reasoning", detail["verifier_input"])
        self.assertEqual(self.path.read_bytes(), before)

    def test_saved_input_remains_authoritative(self) -> None:
        detail = self.detail(2)
        self.assertEqual(detail["verifier_input"], "authoritative saved prompt")
        self.assertEqual(detail["verifier_input_source"], "saved")

    @requires_browsecomp_gates
    def test_historical_format_failure_is_shown_as_reconstructed_skip(self) -> None:
        detail = self.detail(1)
        self.assertIsNone(detail["verifier_input"])
        self.assertEqual(detail["verifier_skipped_reason"], "format:think_close_count:0")
        self.assertEqual(detail["verifier_skipped_reason_source"], "reconstructed")

    def test_historical_incomplete_rollout_is_shown_as_reconstructed_skip(self) -> None:
        detail = self.detail(3)
        self.assertIsNone(detail["verifier_input"])
        self.assertEqual(detail["verifier_skipped_reason"], "termination_reason:response_limit")
        self.assertEqual(detail["verifier_skipped_reason_source"], "reconstructed")

    def test_question_is_recovered_from_the_last_user_turn(self) -> None:
        prompt = (
            "<|im_start|>system\nsystem<|im_end|><|im_start|>user\nfirst<|im_end|><|im_start|>user\nsecond<|im_end|>"
        )
        self.assertEqual(question_from_prompt(prompt), "second")

    @requires_browsecomp_gates
    def test_judge_consistency_flags_use_only_the_judge_visible_response(self) -> None:
        self.assertEqual(
            self.store.query(run=self.logical_run, step=4, category="judge_negative_has_answer")["total"], 0
        )
        self.assertEqual(
            self.store.query(run=self.logical_run, step=5, category="judge_negative_has_answer")["total"], 1
        )
        self.assertEqual(
            self.store.query(run=self.logical_run, step=6, category="judge_positive_no_answer")["total"], 1
        )

    @requires_browsecomp_gates
    def test_format_error_uses_complete_irregular_terminal_turn(self) -> None:
        result = self.store.query(run=self.logical_run, step=7, category="format_error")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["records"][0]["format_error_reason"], "think_close_count:2")
        self.assertIn("first</think>second</think>Alpha", result["records"][0]["terminal_preview"])


class OutcomeClassificationTest(unittest.TestCase):
    """Zero reward means very different things depending on whether the judge ran."""

    def test_judged_outcomes_split_on_reward(self) -> None:
        self.assertIsNone(incomplete_reason("Answer: Alpha", "generation_complete", "stop"))

    def test_length_and_step_caps_never_reach_the_judge(self) -> None:
        for reason in ("response_limit", "max_steps", "context_limit", "reset_failure", "generation_failure"):
            self.assertEqual(incomplete_reason("Answer: Alpha", reason, "stop"), reason)

    def test_missing_terminal_message_never_reaches_the_judge(self) -> None:
        self.assertEqual(incomplete_reason("   ", "generation_complete", "stop"), "no_terminal_message")

    def test_unclean_final_generation_never_reaches_the_judge(self) -> None:
        self.assertEqual(incomplete_reason("Answer: Alpha", "generation_complete", "length"), "unclean_stop")


class GroupAnnotationTest(unittest.TestCase):
    def annotate(self, rewards, key="prompt_id"):
        records = [
            {
                "prompt_id": f"p{index // len(rewards)}",
                "prompt_idx": 0,
                "reward": value,
                "categories": [],
                "reasons": [],
            }
            for index, value in enumerate(rewards)
        ]
        annotate_groups(records)
        return records

    def test_all_wrong_group_is_labelled(self) -> None:
        records = self.annotate([0.0, 0.0, 0.0, 0.0])
        self.assertTrue(all(r["group_shape"] == "all_wrong_group" for r in records))
        self.assertTrue(all("all_wrong_group" in r["categories"] for r in records))
        self.assertEqual(records[0]["group_correct"], 0)

    def test_all_correct_group_is_labelled(self) -> None:
        records = self.annotate([1.0, 1.0, 1.0])
        self.assertTrue(all(r["group_shape"] == "all_correct_group" for r in records))
        self.assertEqual(records[0]["group_correct"], 3)

    def test_mixed_group_records_the_split(self) -> None:
        records = self.annotate([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(records[0]["group_shape"], "mixed_group")
        self.assertEqual((records[0]["group_correct"], records[0]["group_size"]), (1, 10))
        self.assertEqual(records[0]["group_difficulty"], "hard_group")

    def test_mixed_groups_are_split_into_hard_learning_and_easy_bands(self) -> None:
        hard = self.annotate([1.0] + [0.0] * 7)
        learning = self.annotate([1.0] * 4 + [0.0] * 4)
        easy = self.annotate([1.0] * 7 + [0.0])

        self.assertEqual(hard[0]["group_difficulty"], "hard_group")
        self.assertEqual(learning[0]["group_difficulty"], "learning_group")
        self.assertEqual(easy[0]["group_difficulty"], "easy_group")
        self.assertEqual(easy[0]["group_pass_rate"], 0.875)

    def test_lopsided_mixed_group_is_a_near_miss(self) -> None:
        near = self.annotate([1.0] + [0.0] * 9)
        self.assertIn("near_miss_group", near[0]["categories"])
        balanced = self.annotate([1.0] * 5 + [0.0] * 5)
        self.assertNotIn("near_miss_group", balanced[0]["categories"])

    def test_groups_are_keyed_independently(self) -> None:
        records = [
            {"prompt_id": "a", "prompt_idx": 0, "reward": 1.0, "categories": [], "reasons": []},
            {"prompt_id": "b", "prompt_idx": 0, "reward": 0.0, "categories": [], "reasons": []},
        ]
        annotate_groups(records)
        self.assertEqual(records[0]["group_shape"], "all_correct_group")
        self.assertEqual(records[1]["group_shape"], "all_wrong_group")

    def test_prompt_idx_is_the_fallback_key(self) -> None:
        records = [
            {"prompt_idx": 7, "reward": 1.0, "categories": [], "reasons": []},
            {"prompt_idx": 7, "reward": 0.0, "categories": [], "reasons": []},
        ]
        annotate_groups(records)
        self.assertEqual(records[0]["group_key"], "7")
        self.assertEqual(records[0]["group_size"], 2)

    def test_repeated_filtered_prompt_ids_remain_separate_groups(self) -> None:
        records = [
            {
                "source": "filtered",
                "prompt_id": "same",
                "prompt_idx": 0,
                "sample_idx": sample_idx,
                "reward": reward,
                "categories": [],
                "reasons": [],
            }
            for sample_idx, reward in ((0, 0.0), (1, 0.0), (0, 1.0), (1, 1.0))
        ]
        annotate_groups(records)

        self.assertEqual([record["group_key"] for record in records], ["same#1", "same#1", "same#2", "same#2"])
        self.assertEqual(records[0]["group_difficulty"], "all_wrong_group")
        self.assertEqual(records[2]["group_difficulty"], "all_correct_group")


class SegmentTrajectoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = segment_trajectory(TRAJECTORY)

    def test_segments_are_ordered_by_kind_and_role(self) -> None:
        self.assertEqual(
            [(segment["role"], segment["kind"]) for segment in self.segments],
            [
                ("assistant", "reasoning"),
                ("assistant", "tool_call"),
                ("assistant", "tool_call"),
                ("user", "tool_result"),
                ("user", "tool_result"),
                ("assistant", "reasoning"),
                ("assistant", "assistant_text"),
            ],
        )

    def test_unopened_leading_reasoning_block_is_captured(self) -> None:
        self.assertEqual(self.segments[0]["content"], "Let me plan the search.")

    def test_tool_names_are_extracted_per_call(self) -> None:
        names = [segment.get("tool_name") for segment in self.segments if segment["kind"] == "tool_call"]
        self.assertEqual(names, ["search", "visit"])

    def test_observations_and_final_answer_are_separated(self) -> None:
        results = [segment["content"] for segment in self.segments if segment["kind"] == "tool_result"]
        self.assertEqual(results, ["first observation", "second observation"])
        self.assertEqual(self.segments[-1]["content"], "Answer: **Alpha**")

    def test_chat_markers_are_stripped_and_indices_are_dense(self) -> None:
        self.assertTrue(all("<|im_" not in segment["content"] for segment in self.segments))
        self.assertEqual([segment["index"] for segment in self.segments], list(range(len(self.segments))))
        self.assertTrue(all(segment["char_len"] == len(segment["content"]) for segment in self.segments))

    def test_plain_text_without_markers_is_a_single_assistant_segment(self) -> None:
        segments = segment_trajectory("just prose")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["kind"], "assistant_text")

    def test_dangling_think_tag_becomes_reasoning_not_a_response(self) -> None:
        segments = segment_trajectory("wrapping up<|im_end|><|im_start|>assistant\n<think>\nstill thinking")
        self.assertEqual(
            [(segment["kind"], segment["content"]) for segment in segments],
            [("assistant_text", "wrapping up"), ("reasoning", "still thinking")],
        )

    def test_bare_dangling_think_tag_is_dropped(self) -> None:
        self.assertEqual(segment_trajectory("<|im_start|>assistant\n<think>"), [])


class RecursiveDiscoveryTest(unittest.TestCase):
    """One root may hold many training runs, each in its own subdirectory."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for index, name in enumerate(("alpha__42__1", "beta__42__2")):
            run_dir = self.root / f"training_{name}"
            (run_dir / "filtered").mkdir(parents=True)
            accepted = run_dir / f"{name}_rollouts_000000.jsonl"
            accepted.write_text(json.dumps(record(index, 0, reward=1, terminal="Answer: Alpha")) + "\n")
            filtered = run_dir / "filtered" / f"{name}_filtered_rollouts_000000.jsonl"
            filtered.write_text(json.dumps(record(index, 1, reward=0, terminal="No answer", filtered=True)) + "\n")
            (run_dir / f"{name}_metadata.jsonl").write_text(
                json.dumps({"run_name": name, "git_commit": f"sha{index}", "model_name": "local-test"}) + "\n"
            )
        self.store = RolloutStore(self.root, response_limit=8)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_every_nested_run_is_discovered_with_both_sources(self) -> None:
        meta = self.store.meta()
        self.assertEqual([run["name"] for run in meta["runs"]], ["alpha__42__1", "beta__42__2"])
        for run in meta["runs"]:
            self.assertEqual(run["accepted_files"], 1)
            self.assertEqual(run["filtered_files"], 1)
            self.assertEqual(run["metadata"]["model_name"], "local-test")

    def test_runs_stay_isolated_when_querying_one_of_them(self) -> None:
        result = self.store.query(run="beta__42__2", step=1, source="accepted", category="all")
        self.assertEqual(result["total"], 1)
        self.assertEqual(self.store.query(run="alpha__42__1", step=1, source="accepted", category="all")["total"], 0)


if __name__ == "__main__":
    unittest.main()
