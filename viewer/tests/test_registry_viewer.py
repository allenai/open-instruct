from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

from viewer.experiment_service import ExperimentService
from viewer.registry_index import RegistryIndex
from viewer.rollout_store import RolloutStore
from viewer.training_registry import TrainingRegistry, TrainingRegistryError


def rollout_record(step: int, sample_idx: int, reward: float) -> dict[str, Any]:
    return {
        "step": step,
        "sample_idx": sample_idx,
        "prompt_idx": 0,
        "prompt_tokens": [1],
        "response_tokens": [1, 2],
        "reward": reward,
        "advantage": reward - 0.5,
        "finish_reason": "stop",
        "dataset": ["re_search"],
        "ground_truth": ["Alpha"],
        "request_info": {
            "num_calls": 0,
            "timeouts": 0,
            "tool_errors": "",
            "tool_call_stats": [],
            "rollout_state": {
                "terminal_model_text": "Answer: Alpha",
                "termination_reason": "generation_complete",
                "generation_finish_reason": "stop",
            },
        },
    }


def write_attempt(directory: Path, attempt: str, rows: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{attempt}_rollouts_000000.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    metadata = {
        "run_name": attempt,
        "git_commit": "abc123",
        "model_name": "local-test-tokenizer",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    (directory / f"{attempt}_metadata.jsonl").write_text(json.dumps(metadata) + "\n", encoding="utf-8")


def write_registry(repo_root: Path, *, attempts: list[str]) -> TrainingRegistry:
    registry_dir = repo_root / "viewer" / "registry"
    training_dir = registry_dir / "trainings"
    training_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "kind": "training_registry",
        "repo_root": "../..",
        "defaults": {"wandb": {"entity": "entity", "project": "project", "validation_metric": "eval/score"}},
    }
    training = {
        "schema_version": 1,
        "kind": "training",
        "id": "registered-training",
        "title": "Registered training",
        "classification": "evaluated",
        "visibility": "default",
        "tags": {"model": "Qwen3.5-4B", "corpus": "72k"},
        "wandb": {"run_id": "wandb-registered", "rollout_artifact_offset": -1},
        "launches": [
            {
                "id": "initial",
                "relation": "initial",
                "script": "scripts/train.sh",
                "git_commit": "abc123",
                "beaker_experiment": "01EXPERIMENT",
                "image": "01IMAGE",
                "rollouts": [{"path": "rl_rollouts/registered", "attempts": attempts}],
            }
        ],
        "artifacts": {
            "furthest_step": 12,
            "best_evaluation": {
                "benchmark": "browsecomp-plus-bm25-830",
                "step": 10,
                "correct": 415,
                "total": 830,
                "checkpoint": {"step": 10, "path": "checkpoints/step_10"},
            },
            "latest_checkpoint": {"step": 12, "path": "checkpoints/step_12"},
        },
    }
    (registry_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (training_dir / "registered-training.yaml").write_text(yaml.safe_dump(training, sort_keys=False), encoding="utf-8")
    return TrainingRegistry(registry_dir)


class RegistryRolloutStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temporary.name)
        self.rollout_root = self.repo_root / "rl_rollouts"
        self.registered = self.rollout_root / "registered"
        # Deliberately put the larger timestamp first. Registry order, not a
        # timestamp-name heuristic, declares the later restart.
        self.old_attempt = "experiment__42__999"
        self.new_attempt = "experiment__42__100"
        write_attempt(self.registered, self.old_attempt, [rollout_record(0, 1, 0), rollout_record(1, 1, 0)])
        write_attempt(self.registered, self.new_attempt, [rollout_record(1, 9, 1), rollout_record(2, 9, 1)])
        write_attempt(self.rollout_root / "unregistered", "unregistered__42__123", [rollout_record(5, 55, 1)])
        self.registry = write_registry(self.repo_root, attempts=[self.old_attempt, self.new_attempt])
        self.index = RegistryIndex(self.registry)
        self.store = RolloutStore(self.rollout_root, eval_index=self.index)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_explicit_registry_groups_restarts_and_filters_unregistered_files(self) -> None:
        meta = self.store.meta()
        self.assertEqual(len(meta["runs"]), 1)
        run = meta["runs"][0]
        self.assertEqual(run["name"], "training:registered-training")
        self.assertEqual(run["registry_id"], "registered-training")
        self.assertEqual(run["attempts"], [self.old_attempt, self.new_attempt])
        self.assertEqual(run["accepted_files"], 2)
        self.assertNotIn("unregistered__42__123", self.store._attempt_to_run)
        self.assertTrue(all("unregistered" not in str(item.path) for item in self.store.files))

    def test_registry_precedence_wins_for_overlapping_restart_steps(self) -> None:
        logical_run = "training:registered-training"
        self.assertEqual(self.store.steps(logical_run)["steps"], [0, 1, 2])
        overlap = self.store.query(run=logical_run, step=1, category="all")
        self.assertEqual(overlap["stats"]["records"], 1)
        self.assertEqual(overlap["records"][0]["sample_idx"], 9)


class _SegmentedWandbIndex:
    def __init__(self) -> None:
        self.invalidated: list[str] = []
        self.statuses = {
            "wandb-registered": {"state": "fresh", "fetched_at": "fetched-wandb-registered", "error": None},
            "wandb-continuation": {"state": "fresh", "fetched_at": "fetched-wandb-continuation", "error": None},
        }

    def evaluations(self, run_name, directory, *, run_id, metric, artifact_step_offset):
        del run_name, directory, metric
        values = {"wandb-registered": [(10, 0.4), (12, 0.5)], "wandb-continuation": [(12, 0.55), (14, 0.6)]}
        return [
            {
                "artifact_step": step + artifact_step_offset,
                "optimizer_step": step,
                "score": score,
                "metric": "eval/score",
                "wandb_run_id": run_id,
            }
            for step, score in values[run_id]
        ]

    def training_metrics(self, run_id):
        values = {"wandb-registered": [(10, 0.4), (12, 0.5)], "wandb-continuation": [(12, 0.55), (14, 0.6)]}
        return {
            "series": {
                "reward": {
                    "metric": "scores",
                    "points": [{"optimizer_step": step, "value": value} for step, value in values[run_id]],
                }
            }
        }

    def progress(self, run_id):
        return {
            "optimizer_step": {"wandb-registered": 13, "wandb-continuation": 12}[run_id],
            "run_state": {"wandb-registered": "failed", "wandb-continuation": "running"}[run_id],
        }

    def status(self, run_id):
        return self.statuses[run_id]

    def invalidate(self, run_id, *, include_training_metrics=True):
        del include_training_metrics
        self.invalidated.append(run_id)


class SegmentedWandbLineageTest(unittest.TestCase):
    def test_launch_specific_wandb_runs_are_stitched_in_launch_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            rollout_dir = repo_root / "rl_rollouts" / "registered"
            write_attempt(rollout_dir, "attempt-one", [rollout_record(9, 1, 0)])
            write_attempt(rollout_dir, "attempt-two", [rollout_record(13, 2, 1)])
            registry = write_registry(repo_root, attempts=["attempt-one"])
            registry_path = repo_root / "viewer" / "registry" / "trainings" / "registered-training.yaml"
            payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
            payload["launches"].append(
                {
                    "id": "checkpoint-resume",
                    "relation": "checkpoint-resume",
                    "wandb": {"run_id": "wandb-continuation"},
                    "rollouts": [{"path": "rl_rollouts/registered", "attempts": ["attempt-two"]}],
                }
            )
            registry_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            registry.refresh()
            index = RegistryIndex(registry)
            fake = _SegmentedWandbIndex()
            index._wandb["entity/project"] = fake  # type: ignore[assignment]

            self.assertEqual(index.lineage("attempt-one", "registered")["wandb_run_id"], "wandb-registered")
            self.assertEqual(index.lineage("attempt-two", "registered")["wandb_run_id"], "wandb-continuation")
            evaluations = index.validation_evaluations("registered-training")
            self.assertEqual(
                [(row["optimizer_step"], row["score"], row["wandb_run_id"]) for row in evaluations],
                [(10, 0.4, "wandb-registered"), (12, 0.55, "wandb-continuation"), (14, 0.6, "wandb-continuation")],
            )
            rollout_evaluations = index.evaluations("attempt-two", "registered", run_id="wandb-continuation")
            self.assertEqual(
                [(row["optimizer_step"], row["wandb_run_id"]) for row in rollout_evaluations],
                [(10, "wandb-registered"), (12, "wandb-continuation"), (14, "wandb-continuation")],
            )
            points = index.training_metrics("registered-training")["series"]["reward"]["points"]
            self.assertEqual(
                points,
                [
                    {"optimizer_step": 10, "value": 0.4},
                    {"optimizer_step": 12, "value": 0.55},
                    {"optimizer_step": 14, "value": 0.6},
                ],
            )
            progress = index.training_progress("registered-training")
            self.assertEqual(progress["optimizer_step"], 13)
            self.assertEqual(progress["run_state"], "running")
            self.assertEqual(progress["wandb_run_id"], "wandb-continuation")
            self.assertEqual(index.validation_status("registered-training")["wandb_run_id"], "wandb-continuation")
            fake.statuses["wandb-registered"] = {
                "state": "error",
                "fetched_at": "failed-fetch",
                "error": "source unavailable",
            }
            status = index.validation_status("registered-training")
            self.assertEqual(status["state"], "error")
            self.assertIn("wandb-registered: source unavailable", status["error"])
            index.invalidate("registered-training")
            self.assertEqual(fake.invalidated, ["wandb-registered", "wandb-continuation"])


class FakeRegistryIndex:
    def __init__(self) -> None:
        self.invalidated: list[tuple[str | None, bool]] = []
        self.refreshed = False

    def refresh_registry(self) -> None:
        self.refreshed = True

    def invalidate(self, training_id: str | None = None, *, include_training_metrics: bool = True) -> None:
        self.invalidated.append((training_id, include_training_metrics))

    def validation_evaluations(self, training_id: str) -> list[dict[str, Any]]:
        assert training_id == "registered-training"
        return [
            {
                "artifact_step": 9,
                "optimizer_step": 10,
                "score": 0.4,
                "metric": "eval/score",
                "wandb_run_id": "wandb-registered",
            },
            {
                "artifact_step": 11,
                "optimizer_step": 12,
                "score": 0.5,
                "metric": "eval/score",
                "wandb_run_id": "wandb-registered",
            },
        ]

    def validation_status(self, training_id: str) -> dict[str, Any]:
        assert training_id == "registered-training"
        return {"state": "fresh", "fetched_at": "2026-08-11T00:00:00+00:00", "error": None}

    def training_metrics(self, training_id: str) -> dict[str, Any]:
        assert training_id == "registered-training"
        return {"series": {"reward": {"metric": "scores", "points": [{"optimizer_step": 12, "value": 0.6}]}}}

    def training_progress(self, training_id: str) -> dict[str, Any]:
        assert training_id == "registered-training"
        return {"optimizer_step": 20, "run_state": "running"}


class BlockingFakeRegistryIndex(FakeRegistryIndex):
    def __init__(self) -> None:
        super().__init__()
        self.block = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def validation_evaluations(self, training_id: str) -> list[dict[str, Any]]:
        if self.block:
            self.entered.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("test did not release the W&B refresh")
        return super().validation_evaluations(training_id)


class ExperimentServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temporary.name)
        rollout_root = self.repo_root / "rl_rollouts"
        rollout_dir = rollout_root / "registered"
        self.attempt = "experiment__42__123"
        write_attempt(rollout_dir, self.attempt, [rollout_record(0, 0, 1)])
        script = self.repo_root / "scripts" / "train.sh"
        script.parent.mkdir(parents=True)
        script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        for step in (10, 12):
            checkpoint = self.repo_root / "checkpoints" / f"step_{step}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
        self.registry = write_registry(self.repo_root, attempts=[self.attempt])
        real_index = RegistryIndex(self.registry)
        self.store = RolloutStore(rollout_root, eval_index=real_index)
        self.fake_index = FakeRegistryIndex()
        self.service = ExperimentService(self.registry, self.store, self.fake_index)  # type: ignore[arg-type]
        self.service._refresh_validations(list(self.registry.trainings), force=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_catalog_joins_registry_filesystem_rollouts_and_validation_snapshot(self) -> None:
        result = self.service.list_trainings()
        self.assertEqual(result["summary"]["total"], 1)
        self.assertEqual(result["summary"]["classifications"], {"evaluated": 1})
        self.assertEqual(result["summary"]["inspectable_rollouts"], 1)
        training = result["trainings"][0]
        self.assertEqual(training["id"], "registered-training")
        self.assertEqual(training["furthest_step"], 20)
        self.assertTrue(training["launches"][0]["script_exists"])
        self.assertTrue(training["launches"][0]["rollouts"][0]["exists"])
        self.assertTrue(training["latest_checkpoint"]["complete"])
        self.assertEqual(training["live"]["rollouts"]["logical_run"], "training:registered-training")
        validation = training["live"]["validation"]
        self.assertEqual(validation["latest"]["optimizer_step"], 12)
        self.assertEqual(validation["best"]["score"], 0.5)
        self.assertEqual(validation["status"]["state"], "fresh")
        self.assertEqual(self.fake_index.invalidated, [("registered-training", False)])

    def test_local_urls_are_encoded_and_registered_paths_can_be_inspected(self) -> None:
        training = self.service.get_training("registered-training")
        script = training["launches"][0]
        parsed = urlparse(script["script_url"])
        self.assertEqual(parsed.path, "/api/path")
        self.assertEqual(parse_qs(parsed.query)["path"], [script["script_path"]])
        script_info = self.service.path_info(script["script_path"])
        self.assertEqual(script_info["kind"], "file")
        self.assertIn("#!/usr/bin/env bash", script_info["content"])

        metrics = self.service.get_training_metrics("registered-training")
        self.assertEqual(metrics["training_id"], "registered-training")
        self.assertEqual(metrics["series"]["reward"]["points"][0]["value"], 0.6)

        checkpoint = training["latest_checkpoint"]
        checkpoint_info = self.service.path_info(checkpoint["path"])
        self.assertEqual(checkpoint_info["kind"], "directory")
        self.assertEqual(checkpoint_info["entries"][0]["name"], "config.json")

    def test_catalog_refresh_invalidates_chart_histories(self) -> None:
        self.service.refresh_catalog()
        self.assertTrue(self.fake_index.refreshed)
        self.assertEqual(self.fake_index.invalidated[-1], (None, True))

    def test_background_refresh_marks_detail_pending_until_its_new_snapshot_arrives(self) -> None:
        index = BlockingFakeRegistryIndex()
        service = ExperimentService(self.registry, self.store, index)  # type: ignore[arg-type]
        service._refresh_validations(list(self.registry.trainings), force=True)
        index.block = True

        self.assertTrue(service.start_validation_refresh(force=True))
        self.assertTrue(index.entered.wait(timeout=2))
        refreshing = service.get_training("registered-training")
        self.assertEqual(refreshing["furthest_step"], 20)
        self.assertEqual(refreshing["live"]["validation"]["status"]["state"], "refreshing")

        index.release.set()
        assert service._refresh_thread is not None
        service._refresh_thread.join(timeout=2)
        refreshed = service.get_training("registered-training")
        self.assertEqual(refreshed["live"]["validation"]["status"]["state"], "fresh")

    def test_unregistered_paths_cannot_be_read(self) -> None:
        secret = self.repo_root / "not-registered.txt"
        secret.write_text("not exposed", encoding="utf-8")
        with self.assertRaisesRegex(TrainingRegistryError, "Path is not registered"):
            self.service.path_info(str(secret))


if __name__ == "__main__":
    unittest.main()
