from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from viewer.wandb_evals import WandbEvalIndex, artifact_step


class FakeRun:
    def __init__(
        self, run_id, display_name="", run_name="", save_path="", history=None, summary=None, created_at=None
    ):
        self.id = run_id
        self.name = display_name
        self.created_at = created_at
        self.state = "running"
        self.config = {"run_name": run_name, "rollouts_save_path": save_path}
        self.summary = summary if summary is not None else {"eval/objective/verifiable_correct_rate": 0.3}
        self._history = history or []
        self.scan_keys = []

    def scan_history(self, keys=None):
        self.scan_keys.append(tuple(keys or ()))
        if not keys:
            return list(self._history)
        return [{key: row[key] for key in keys} for row in self._history if all(key in row for key in keys)]


class FakeApi:
    def __init__(self, runs):
        self._runs = runs
        self.run_calls = 0

    def runs(self, path):
        return list(self._runs)

    def run(self, path):
        self.run_calls += 1
        return next(r for r in self._runs if path.endswith(r.id))


def index_with(runs, overrides=None):
    index = WandbEvalIndex("entity/project", overrides)
    index._api = FakeApi(runs)
    return index


HISTORY = [
    {"_step": 10, "eval/objective/verifiable_correct_rate": 0.2},
    {"_step": 20, "eval/objective/verifiable_correct_rate": 0.25},
    {"_step": 41, "eval/objective/verifiable_correct_rate": 0.3},
    {"_step": 15, "eval/objective/verifiable_correct_rate": None},
]


class ArtifactStepTest(unittest.TestCase):
    def test_wandb_step_is_one_ahead_of_the_artifact_step(self) -> None:
        # W&B logs the optimizer step; rollouts are stored under the step before it.
        self.assertEqual(artifact_step(1), 0)
        self.assertEqual(artifact_step(140), 139)


class ResolutionTest(unittest.TestCase):
    def test_progress_reads_and_caches_explicit_training_step(self) -> None:
        run = FakeRun("abc", summary={"training_step": 160})
        index = index_with([run])
        self.assertEqual(index.progress("abc"), {"optimizer_step": 160, "run_state": "running"})
        self.assertEqual(index.progress("abc"), {"optimizer_step": 160, "run_state": "running"})
        self.assertEqual(index._api.run_calls, 1)

    def test_progress_api_failure_returns_an_empty_snapshot(self) -> None:
        class Broken:
            def run(self, path):
                raise RuntimeError("no credentials")

        index = WandbEvalIndex("entity/project")
        index._api = Broken()

        self.assertEqual(index.progress("abc"), {"optimizer_step": None, "run_state": None})

    def test_matches_on_config_run_name(self) -> None:
        index = index_with([FakeRun("abc", run_name="exp__42__1", history=HISTORY)])
        self.assertEqual(index.evaluated_steps("exp__42__1", "some_dir"), [9, 19, 40])

    def test_matches_on_display_name(self) -> None:
        index = index_with([FakeRun("abc", display_name="exp__42__1", history=HISTORY)])
        self.assertEqual(index.evaluated_steps("exp__42__1", "some_dir"), [9, 19, 40])

    def test_falls_back_to_the_rollout_directory(self) -> None:
        run = FakeRun("abc", run_name="exp__42__9", save_path="/weka/x/rl_rollouts/pool_a", history=HISTORY)
        index = index_with([run])
        self.assertEqual(index.evaluated_steps("exp__42__1", "pool_a"), [9, 19, 40])

    def test_restart_attempt_resolves_to_one_wandb_lineage(self) -> None:
        run = FakeRun(
            "abc",
            run_name="exp__42__200",
            save_path="/weka/x/rl_rollouts/pool_a",
            history=HISTORY,
            created_at="1970-01-01T00:01:40Z",
        )
        index = index_with([run])
        self.assertEqual(index.lineage("exp__42__100", "pool_a"), {"id": "abc", "name": "exp__42__200"})

    def test_validation_payload_includes_scores_and_both_step_spaces(self) -> None:
        index = index_with([FakeRun("abc", run_name="exp__42__1", history=HISTORY)])
        self.assertEqual(
            index.evaluations("exp__42__1", "some_dir")[0],
            {
                "artifact_step": 9,
                "optimizer_step": 10,
                "score": 0.2,
                "metric": "eval/objective/verifiable_correct_rate",
                "wandb_run_id": "abc",
            },
        )

    def test_delayed_evaluation_prefers_explicit_training_step(self) -> None:
        history = [{"_step": 27, "training_step": 20, "eval/local_bm25/accuracy": 0.4}]
        index = index_with([FakeRun("abc", run_name="exp__42__1", history=history)])
        self.assertEqual(
            index.evaluations("exp__42__1", "some_dir", metric="eval/local_bm25/accuracy")[0]["optimizer_step"], 20
        )

    def test_an_override_wins_over_automatic_matching(self) -> None:
        # Resuming overwrites a W&B run's name and config, so earlier attempts of
        # a lineage can only be reached explicitly.
        runs = [FakeRun("keep", run_name="exp__42__1"), FakeRun("wanted", history=HISTORY)]
        index = index_with(runs, {"exp__42__1": "wanted"})
        self.assertEqual(index.evaluated_steps("exp__42__1", "dir"), [9, 19, 40])

    def test_unmatched_run_reports_no_evaluated_steps(self) -> None:
        index = index_with([FakeRun("abc", run_name="different", save_path="/weka/x/pool_b")])
        self.assertEqual(index.evaluated_steps("exp__42__1", "pool_a"), [])

    def test_results_are_cached_per_run(self) -> None:
        index = index_with([FakeRun("abc", run_name="exp__42__1", history=HISTORY)])
        index.evaluated_steps("exp__42__1", "dir")
        index.evaluated_steps("exp__42__1", "dir")
        self.assertEqual(index._api.run_calls, 1)

    def test_a_broken_api_degrades_to_no_steps(self) -> None:
        class Broken:
            def runs(self, path):
                raise RuntimeError("no credentials")

        index = WandbEvalIndex("entity/project")
        index._api = Broken()
        self.assertEqual(index.evaluated_steps("exp__42__1", "dir"), [])

    def test_metric_choice_prefers_the_verifier_rate(self) -> None:
        run = FakeRun("abc", run_name="exp", summary={"eval/scores": 1, "eval/objective/verifiable_correct_rate": 0.4})
        self.assertEqual(WandbEvalIndex._pick_metric(run), "eval/objective/verifiable_correct_rate")

    def test_metric_choice_accepts_any_eval_key(self) -> None:
        run = FakeRun("abc", run_name="exp", summary={"train/loss": 1, "eval/custom_metric": 2})
        self.assertEqual(WandbEvalIndex._pick_metric(run), "eval/custom_metric")

    def test_runs_without_eval_metrics_report_nothing(self) -> None:
        run = FakeRun("abc", run_name="exp", summary={"train/loss": 1})
        self.assertIsNone(WandbEvalIndex._pick_metric(run))

    def test_training_metrics_reads_compact_scalar_histories(self) -> None:
        history = [
            {
                "_step": 1,
                "training_step": 1,
                "scores": 0.25,
                "val/avg_group_performance_pre_filter": 0.2,
                "val/avg_group_performance_post_filter": 0.25,
                "val/sequence_lengths": 1200,
                "val/terminal_turn_lengths": 320,
                "val/truncated_completion_fraction": 0.1,
                "tools/aggregate/avg_calls_per_rollout": 3.5,
                "tools/search/failure_rate": 0.05,
                "tools/visit/failure_rate": 0.25,
                "format/incomplete_pre_filtering": 0.2,
                "format/terminal_format_pre_filtering": 0.1,
                "format/trajectory_format_pre_filtering": 0.05,
                "format/incomplete_post_filtering": 0.125,
                "format/terminal_format_post_filtering": 0.0625,
                "format/trajectory_format_post_filtering": 0.0,
                "policy/entropy_avg": 0.8,
                "batch/filtered_prompts": 4,
                "batch/filtered_prompts_zero": 3,
                "batch/filtered_prompts_solved": 1,
                "batch/total_prompts": 8,
            },
            {
                "_step": 2,
                "training_step": 2,
                "scores": 0.5,
                "val/avg_group_performance_pre_filter": 0.35,
                "val/avg_group_performance_post_filter": 0.5,
                "val/sequence_lengths": 900,
                "val/terminal_turn_lengths": 240,
                "tools/aggregate/avg_calls_per_rollout": 2.0,
                "tools/search/failure_rate": 0.025,
                "tools/visit/failure_rate": 0.125,
                "format/incomplete_pre_filtering": 0.15,
                "format/terminal_format_pre_filtering": 0.08,
                "format/trajectory_format_pre_filtering": 0.04,
                "format/incomplete_post_filtering": 0.1,
                "format/terminal_format_post_filtering": 0.05,
                "format/trajectory_format_post_filtering": 0.01,
                "policy/entropy_avg": 0.7,
                "batch/filtered_prompts": 8,
                "batch/filtered_prompts_zero": 2,
                "batch/filtered_prompts_solved": 6,
                "batch/total_prompts": 8,
            },
        ]
        summary = {key: value for key, value in history[-1].items() if key not in {"_step", "training_step"}}
        run = FakeRun("abc", history=history, summary=summary)
        index = index_with([run])
        payload = index.training_metrics("abc")["series"]
        self.assertEqual(payload["reward"]["metric"], "scores")
        self.assertEqual(payload["reward"]["points"][-1], {"optimizer_step": 2, "value": 0.5})
        self.assertEqual(payload["group_pass_rate_all"]["metric"], "val/avg_group_performance_pre_filter")
        self.assertEqual(payload["group_pass_rate_all"]["points"][-1]["value"], 0.35)
        self.assertEqual(payload["group_pass_rate_post_mask"]["metric"], "val/avg_group_performance_post_filter")
        self.assertEqual(payload["group_pass_rate_post_mask"]["points"][-1]["value"], 0.5)
        self.assertEqual(payload["length"]["points"][0]["value"], 1200.0)
        self.assertEqual(payload["terminal_length"]["metric"], "val/terminal_turn_lengths")
        self.assertEqual(payload["terminal_length"]["points"][-1]["value"], 240.0)
        self.assertEqual(payload["tool_calls"]["points"][-1]["value"], 2.0)
        self.assertEqual(payload["search_failure_rate"]["points"][-1]["value"], 0.025)
        self.assertEqual(payload["visit_failure_rate"]["points"][-1]["value"], 0.125)
        self.assertEqual(payload["format_incomplete_pre"]["metric"], "format/incomplete_pre_filtering")
        self.assertEqual(payload["format_incomplete_pre"]["points"][-1]["value"], 0.15)
        self.assertEqual(payload["format_terminal_pre"]["points"][-1]["value"], 0.08)
        self.assertEqual(payload["format_trajectory_pre"]["points"][-1]["value"], 0.04)
        self.assertEqual(payload["format_incomplete_post"]["points"][-1]["value"], 0.1)
        self.assertEqual(payload["format_terminal_post"]["points"][-1]["value"], 0.05)
        self.assertEqual(payload["format_trajectory_post"]["points"][-1]["value"], 0.01)
        self.assertEqual(payload["logprob"]["metric"], "policy/entropy_avg")
        self.assertEqual(payload["rejected_group_rate"]["points"][-1]["value"], 0.5)
        self.assertEqual(payload["rejected_all_zero_rate"]["points"][-1]["value"], 0.25)
        self.assertEqual(payload["rejected_all_one_rate"]["points"][-1]["value"], 0.75)
        self.assertIn(("_step", "training_step", "val/terminal_turn_lengths"), run.scan_keys)
        self.assertIn(("_step", "training_step", "val/avg_group_performance_post_filter"), run.scan_keys)

    def test_training_metrics_ignores_undefined_terminal_length(self) -> None:
        run = FakeRun(
            "abc",
            history=[
                {"_step": 1, "training_step": 1, "val/terminal_turn_lengths": 320.0},
                {"_step": 2, "training_step": 2, "val/terminal_turn_lengths": float("nan")},
            ],
            summary={"val/terminal_turn_lengths": float("nan")},
        )

        payload = index_with([run]).training_metrics("abc")["series"]["terminal_length"]

        self.assertEqual(payload["points"], [{"optimizer_step": 1, "value": 320.0}])

    def test_other_histories_do_not_mark_validation_as_fetched(self) -> None:
        metric = "eval/objective/verifiable_correct_rate"
        history = [{"_step": 1, "training_step": 1, "scores": 0.5, metric: 0.25}]
        index = index_with([FakeRun("abc", history=history, summary={"scores": 0.5, metric: 0.25})])

        index.training_metrics("abc")
        self.assertEqual(index.status("abc")["state"], "pending")
        index.progress("abc")
        self.assertEqual(index.status("abc")["state"], "pending")

        index.evaluations("experiment", "rollouts", run_id="abc")
        self.assertEqual(index.status("abc")["state"], "fresh")

    def test_training_metrics_do_not_require_summary_keys(self) -> None:
        run = FakeRun(
            "abc",
            history=[{"_step": 9, "training_step": 42, "scores": 0.625}],
            summary={"eval/local_bm25/accuracy": 0.3},
        )
        payload = index_with([run]).training_metrics("abc")["series"]
        self.assertEqual(payload["reward"]["metric"], "scores")
        self.assertEqual(payload["reward"]["points"], [{"optimizer_step": 42, "value": 0.625}])

    def test_training_metrics_fall_back_to_internal_step(self) -> None:
        run = FakeRun("abc", history=[{"_step": 7, "scores": 0.5}], summary={})
        payload = index_with([run]).training_metrics("abc")["series"]
        self.assertEqual(payload["reward"]["points"], [{"optimizer_step": 7, "value": 0.5}])

    def test_training_metrics_prefers_logprob_drift_and_caches(self) -> None:
        history = [{"_step": 4, "debug/vllm_vs_local_logprob_diff_mean": 0.012, "policy/entropy_avg": 0.9}]
        run = FakeRun("abc", history=history, summary=history[0])
        index = index_with([run])
        first = index.training_metrics("abc")
        second = index.training_metrics("abc")
        self.assertEqual(first, second)
        self.assertEqual(first["series"]["logprob"]["metric"], "debug/vllm_vs_local_logprob_diff_mean")
        index.invalidate("abc", include_training_metrics=False)
        self.assertEqual(index.training_metrics("abc"), first)
        self.assertEqual(index._api.run_calls, 1)

    def test_concurrent_training_metric_requests_share_one_fetch(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class BlockingRun(FakeRun):
            def scan_history(self, keys=None):
                if not started.is_set():
                    started.set()
                    release.wait(timeout=2)
                return super().scan_history(keys)

        run = BlockingRun("abc", history=[{"_step": 4, "scores": 0.5}], summary={})
        index = index_with([run])
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(index.training_metrics, "abc")
            self.assertTrue(started.wait(timeout=1))
            second = executor.submit(index.training_metrics, "abc")
            release.set()
            self.assertEqual(first.result(timeout=2), second.result(timeout=2))
        self.assertEqual(index._api.run_calls, 1)


if __name__ == "__main__":
    unittest.main()
