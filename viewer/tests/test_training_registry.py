from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from viewer.training_registry import TrainingRegistry, TrainingRegistryError

REPO_ROOT = Path(__file__).resolve().parents[2]


def training_row(training_id: str = "training-one", attempt: str = "attempt-one") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "training",
        "id": training_id,
        "title": f"Training {training_id}",
        "classification": "evaluated",
        "visibility": "default",
        "tags": {"model": "Qwen3.5-4B", "corpus": "BM25 72k"},
        "wandb": {"run_id": f"wandb-{training_id}", "rollout_artifact_offset": -2},
        "launches": [
            {
                "id": "initial",
                "relation": "initial",
                "script": "scripts/train.sh",
                "git_commit": "abc123",
                "git_repository": "wu-ming233/open-instruct",
                "beaker_experiment": "01EXPERIMENT",
                "image": "01IMAGE",
                "rollouts": [{"path": f"rl_rollouts/{training_id}", "attempts": [attempt]}],
            }
        ],
        "artifacts": {
            "furthest_step": 12,
            "evaluations": [
                {"benchmark": "browsecomp-plus-bm25-830", "step": 10, "correct": 415, "total": 830},
                {
                    "benchmark": "browsecomp-serper-jina-100",
                    "step": 12,
                    "correct": 44,
                    "total": 100,
                    "checkpoint": {"step": 12, "path": "checkpoints/step_12"},
                },
            ],
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


def config_payload(*, schema_version: int = 1, kind: str = "training_registry") -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "kind": kind,
        "repo_root": "../..",
        "defaults": {
            "wandb": {"entity": "test-entity", "project": "test-project", "validation_metric": "eval/test-score"}
        },
    }


class SplitTrainingRegistryFixtureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temporary.name)
        self.registry_dir = self.repo_root / "viewer" / "registry"
        self.training_dir = self.registry_dir / "trainings"
        self.training_dir.mkdir(parents=True)
        self.write_config()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_config(self, payload: dict[str, Any] | None = None) -> Path:
        path = self.registry_dir / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload or config_payload(), sort_keys=False), encoding="utf-8")
        return path

    def write_training(self, row: dict[str, Any], *, filename: str | None = None) -> Path:
        path = self.training_dir / (filename or f"{row['id']}.yaml")
        path.write_text(yaml.safe_dump(row, sort_keys=False), encoding="utf-8")
        return path

    def create_fixture_artifacts(self) -> None:
        script = self.repo_root / "scripts" / "train.sh"
        script.parent.mkdir()
        script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

        rollouts = self.repo_root / "rl_rollouts" / "training-one"
        rollouts.mkdir(parents=True)
        (rollouts / "attempt-one_metadata.jsonl").write_text("{}\n", encoding="utf-8")

        complete_checkpoint = self.repo_root / "checkpoints" / "step_10"
        complete_checkpoint.mkdir(parents=True)
        (complete_checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
        (self.repo_root / "checkpoints" / "step_12").mkdir(parents=True)

    def test_scans_all_entry_files_in_filename_order_and_resolves_paths(self) -> None:
        self.create_fixture_artifacts()
        self.write_training(training_row("z-training", "attempt-z"))
        self.write_training(training_row())

        registry = TrainingRegistry(self.registry_dir)

        self.assertEqual(registry.root, self.registry_dir.resolve())
        self.assertEqual(registry.path, (self.registry_dir / "config.yaml").resolve())
        self.assertEqual(registry.repo_root, self.repo_root.resolve())
        self.assertEqual([training.id for training in registry.trainings], ["training-one", "z-training"])
        training = registry.get("training-one")
        self.assertEqual(training.wandb.project_path, "test-entity/test-project")
        self.assertEqual(training.wandb.validation_metric, "eval/test-score")
        self.assertEqual(training.wandb.rollout_artifact_offset, -2)
        self.assertEqual(training.rollout_attempts, ("attempt-one",))

        public = registry.public()[0]
        self.assertEqual(public["wandb"]["url"], "https://wandb.ai/test-entity/test-project/runs/wandb-training-one")
        launch = public["launches"][0]
        self.assertEqual(launch["script_path"], str((self.repo_root / "scripts" / "train.sh").resolve()))
        self.assertTrue(launch["script_exists"])
        self.assertEqual(launch["git_repository_url"], "https://github.com/wu-ming233/open-instruct")
        self.assertEqual(launch["git_url"], "https://github.com/wu-ming233/open-instruct/commit/abc123")
        rollout = launch["rollouts"][0]
        self.assertEqual(rollout["display_path"], "rl_rollouts/training-one")
        self.assertEqual(rollout["present_attempts"], ["attempt-one"])
        self.assertTrue(rollout["exists"])
        self.assertTrue(public["best_evaluation"]["checkpoint"]["complete"])
        self.assertEqual(
            [(item["benchmark"], item["step"], item["correct"], item["total"]) for item in public["evaluations"]],
            [("browsecomp-plus-bm25-830", 10, 415, 830), ("browsecomp-serper-jina-100", 12, 44, 100)],
        )
        self.assertTrue(public["evaluations"][0]["checkpoint"]["complete"])
        self.assertTrue(public["latest_checkpoint"]["exists"])
        self.assertFalse(public["latest_checkpoint"]["complete"])

    def test_constructor_also_accepts_config_file_for_convenience(self) -> None:
        self.write_training(training_row())
        registry = TrainingRegistry(self.registry_dir / "config.yaml")
        self.assertEqual(registry.root, self.registry_dir.resolve())
        self.assertEqual(registry.path, (self.registry_dir / "config.yaml").resolve())
        self.assertEqual([training.id for training in registry.trainings], ["training-one"])

    def test_launch_repository_defaults_to_allenai_for_legacy_entries(self) -> None:
        row = training_row()
        row["launches"][0].pop("git_repository")
        self.write_training(row)
        launch = TrainingRegistry(self.registry_dir).public()[0]["launches"][0]
        self.assertEqual(launch["git_url"], "https://github.com/allenai/open-instruct/commit/abc123")

    def test_launch_repository_rejects_an_unlinkable_value(self) -> None:
        row = training_row()
        row["launches"][0]["git_repository"] = "not a repository"
        self.write_training(row)
        with self.assertRaisesRegex(TrainingRegistryError, "git_repository"):
            TrainingRegistry(self.registry_dir)

    def test_launch_can_register_a_new_wandb_run_in_the_same_lineage(self) -> None:
        row = training_row()
        row["launches"].append(
            {
                "id": "checkpoint-resume",
                "relation": "checkpoint-resume",
                "wandb": {"run_id": "wandb-continuation"},
                "rollouts": [{"path": "rl_rollouts/training-one", "attempts": ["attempt-two"]}],
            }
        )
        self.write_training(row)

        training = TrainingRegistry(self.registry_dir).get("training-one")

        self.assertEqual(
            [reference.run_id for reference in training.wandb_runs], ["wandb-training-one", "wandb-continuation"]
        )
        self.assertEqual(training.active_wandb.run_id, "wandb-continuation")
        self.assertEqual(training.launches[-1].wandb.project_path, "test-entity/test-project")
        public = training.public(self.repo_root)
        self.assertEqual(public["wandb"]["run_id"], "wandb-continuation")
        self.assertEqual(
            [reference["run_id"] for reference in public["wandb_runs"]], ["wandb-training-one", "wandb-continuation"]
        )
        self.assertEqual(public["launches"][0]["wandb"]["run_id"], "wandb-training-one")
        self.assertEqual(public["launches"][-1]["wandb"]["run_id"], "wandb-continuation")

    def test_latest_launch_use_determines_the_active_wandb_run(self) -> None:
        row = training_row()
        row["launches"].extend(
            [
                {"id": "branch", "wandb": {"run_id": "wandb-branch"}},
                {"id": "return", "wandb": {"run_id": "wandb-training-one"}},
            ]
        )
        self.write_training(row)

        training = TrainingRegistry(self.registry_dir).get("training-one")

        self.assertEqual(
            [reference.run_id for reference in training.wandb_runs], ["wandb-branch", "wandb-training-one"]
        )
        self.assertEqual(training.active_wandb.run_id, "wandb-training-one")

    def test_duplicate_full_evaluation_steps_are_rejected_per_benchmark(self) -> None:
        row = training_row()
        row["artifacts"]["evaluations"].append(
            {"benchmark": "browsecomp-serper-jina-100", "step": 12, "correct": 45, "total": 100}
        )
        self.write_training(row)
        with self.assertRaisesRegex(TrainingRegistryError, "duplicate benchmark/step"):
            TrainingRegistry(self.registry_dir)

    def test_inference_artifact_cannot_belong_to_two_evaluations(self) -> None:
        first = training_row("training-one", "attempt-one")
        second = training_row("training-two", "attempt-two")
        for row in (first, second):
            row["artifacts"]["evaluations"][0]["inference_artifact"] = {
                "path": "output/shared-evaluation",
                "schema": "open_instruct_inference_v1",
                "judged_response_index": -1,
            }
        self.write_training(first)
        self.write_training(second)
        with self.assertRaisesRegex(TrainingRegistryError, "inference artifact .* belongs to both"):
            TrainingRegistry(self.registry_dir)

    def test_inference_artifact_validates_schema_and_response_index(self) -> None:
        for key, value in (("schema", "unknown"), ("judged_response_index", "last")):
            with self.subTest(key=key):
                row = training_row()
                row["artifacts"]["evaluations"][0]["inference_artifact"] = {
                    "path": "output/evaluation",
                    "schema": "open_instruct_inference_v1",
                    "judged_response_index": -1,
                    key: value,
                }
                self.write_training(row)
                with self.assertRaisesRegex(TrainingRegistryError, key):
                    TrainingRegistry(self.registry_dir)

    def test_full_evaluations_can_supply_the_best_when_legacy_best_is_absent(self) -> None:
        row = training_row()
        row["artifacts"].pop("best_evaluation")
        self.write_training(row)
        public = TrainingRegistry(self.registry_dir).public()[0]
        self.assertEqual(public["best_evaluation"]["benchmark"], "browsecomp-plus-bm25-830")
        self.assertEqual(public["best_evaluation"]["step"], 10)

    def test_refresh_discovers_new_entry_files_deterministically(self) -> None:
        self.write_training(training_row("m-training", "attempt-m"))
        registry = TrainingRegistry(self.registry_dir)
        self.assertEqual([training.id for training in registry.trainings], ["m-training"])

        self.write_training(training_row("a-training", "attempt-a"))
        registry.refresh()

        self.assertEqual([training.id for training in registry.trainings], ["a-training", "m-training"])
        self.assertEqual(registry.get("a-training").wandb.run_id, "wandb-a-training")

    def test_filename_must_exactly_match_training_id(self) -> None:
        self.write_training(training_row(), filename="wrong-name.yaml")
        with self.assertRaisesRegex(TrainingRegistryError, "filename.*training-one|training-one.*filename"):
            TrainingRegistry(self.registry_dir)

    def test_config_schema_and_kind_are_required(self) -> None:
        self.write_training(training_row())
        for payload, expected in (
            (config_payload(schema_version=2), "schema_version"),
            (config_payload(kind="training"), "kind"),
        ):
            with self.subTest(payload=payload):
                self.write_config(payload)
                with self.assertRaisesRegex(TrainingRegistryError, expected):
                    TrainingRegistry(self.registry_dir)

    def test_entry_schema_and_kind_are_required(self) -> None:
        for key, value, expected in (("schema_version", 2, "schema_version"), ("kind", "inference", "kind")):
            row = training_row()
            row[key] = value
            self.write_training(row)
            with self.assertRaisesRegex(TrainingRegistryError, expected):
                TrainingRegistry(self.registry_dir)
            (self.training_dir / "training-one.yaml").unlink()

    def test_duplicate_wandb_ids_are_rejected_across_files(self) -> None:
        first = training_row("training-one", "attempt-one")
        second = training_row("training-two", "attempt-two")
        second["wandb"]["run_id"] = first["wandb"]["run_id"]
        self.write_training(first)
        self.write_training(second)
        with self.assertRaisesRegex(TrainingRegistryError, "W&B|wandb|run_id"):
            TrainingRegistry(self.registry_dir)

    def test_launch_wandb_ids_are_rejected_across_training_lineages(self) -> None:
        first = training_row("training-one", "attempt-one")
        first["launches"][0]["wandb"] = {"run_id": "shared-wandb-run"}
        second = training_row("training-two", "attempt-two")
        second["launches"][0]["wandb"] = {"run_id": "shared-wandb-run"}
        self.write_training(first)
        self.write_training(second)

        with self.assertRaisesRegex(TrainingRegistryError, "shared-wandb-run.*both"):
            TrainingRegistry(self.registry_dir)

    def test_rollout_attempt_cannot_belong_to_two_files(self) -> None:
        self.write_training(training_row("training-one", "shared-attempt"))
        self.write_training(training_row("training-two", "shared-attempt"))
        with self.assertRaisesRegex(
            TrainingRegistryError, "rollout attempt 'shared-attempt' belongs to both 'training-one' and 'training-two'"
        ):
            TrainingRegistry(self.registry_dir)

    def test_configured_only_rollout_can_be_missing(self) -> None:
        row = training_row()
        rollout = row["launches"][0]["rollouts"][0]
        rollout["path"] = "rl_rollouts/future-run"
        rollout["attempts"] = []
        rollout["configured_only"] = True
        self.write_training(row)

        status = TrainingRegistry(self.registry_dir).public()[0]["launches"][0]["rollouts"][0]
        self.assertFalse(status["exists"])
        self.assertTrue(status["configured_only"])


class RealTrainingRegistryTest(unittest.TestCase):
    """Integrity checks over the live checked-in registry, independent of its contents."""

    def test_catalog_has_unique_global_identity(self) -> None:
        registry = TrainingRegistry(REPO_ROOT / "viewer" / "registry")

        self.assertGreater(len(registry.trainings), 0)
        wandb_ids = [reference.run_id for training in registry.trainings for reference in training.wandb_runs]
        self.assertEqual(len(wandb_ids), len(set(wandb_ids)))
        attempts = [
            attempt
            for training in registry.trainings
            for launch in training.launches
            for rollout in launch.rollouts
            for attempt in rollout.attempts
        ]
        self.assertEqual(len(attempts), len(set(attempts)))
        expected = [source.stem for source in sorted((REPO_ROOT / "viewer" / "registry" / "trainings").glob("*.yaml"))]
        self.assertEqual([training.id for training in registry.trainings], expected)
        for training in registry.trainings:
            steps = [(item.benchmark.casefold(), item.step) for item in training.evaluations]
            self.assertEqual(len(steps), len(set(steps)), training.id)


if __name__ == "__main__":
    unittest.main()
