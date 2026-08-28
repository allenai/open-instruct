from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from viewer.training_registry import TrainingDefinition, TrainingRegistry, WandbReference
from viewer.wandb_evals import WandbEvalIndex


class RegistryIndex:
    """Resolve exact rollout attempts through the checked-in training registry."""

    registered_only = True

    def __init__(self, registry: TrainingRegistry) -> None:
        self.registry = registry
        self._lock = threading.RLock()
        self._wandb: dict[str, WandbEvalIndex] = {}
        self._attempts: dict[str, tuple[TrainingDefinition, Path, int, WandbReference | None]] = {}
        self._trainings: dict[str, TrainingDefinition] = {}
        self.refresh_registry()

    def refresh_registry(self) -> None:
        self.registry.refresh()
        attempts: dict[str, tuple[TrainingDefinition, Path, int, WandbReference | None]] = {}
        trainings = {training.id: training for training in self.registry.trainings}
        for training in self.registry.trainings:
            precedence = 0
            for launch in training.launches:
                wandb = launch.wandb or training.wandb
                for rollout in launch.rollouts:
                    for attempt in rollout.attempts:
                        attempts[attempt] = (training, rollout.path.resolve(), precedence, wandb)
                        precedence += 1
        with self._lock:
            self._attempts = attempts
            self._trainings = trainings

    def lineage(self, run_name: str, directory: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._attempts.get(run_name)
        if item is None:
            return None
        training, path, _, wandb = item
        # Directory basenames are sufficient here because RolloutStore passes
        # only the containing directory's name. The attempt name remains the
        # globally unique, authoritative key.
        if directory and path.name != directory:
            return None
        return {
            "id": training.id,
            "logical_run": f"training:{training.id}",
            "name": training.title,
            "wandb_run_id": wandb.run_id if wandb else None,
            "registry_id": training.id,
            "classification": training.classification,
            "visibility": training.visibility,
            "tags": training.tags,
        }

    def precedence(self, attempt: str) -> int:
        with self._lock:
            item = self._attempts.get(attempt)
        return item[2] if item else -1

    def evaluations(self, run_name: str, directory: str, *, run_id: str | None = None) -> list[dict[str, Any]]:
        training = self._training_for(run_name, run_id)
        return self._validation_evaluations_for(training, run_name=run_name, directory=directory)

    def invalidate(self, training_id: str | None = None, *, include_training_metrics: bool = True) -> None:
        with self._lock:
            if training_id is None:
                for index in self._wandb.values():
                    index.invalidate(include_training_metrics=include_training_metrics)
                return
            training = self._trainings.get(training_id)
            if training:
                for reference in training.wandb_runs:
                    self._wandb_index(reference.project_path).invalidate(
                        reference.run_id, include_training_metrics=include_training_metrics
                    )

    def training(self, training_id: str) -> TrainingDefinition | None:
        with self._lock:
            return self._trainings.get(training_id)

    def validation_evaluations(self, training_id: str) -> list[dict[str, Any]]:
        training = self.training(training_id)
        return self._validation_evaluations_for(training, run_name=training_id, directory="")

    def _validation_evaluations_for(
        self, training: TrainingDefinition | None, *, run_name: str, directory: str
    ) -> list[dict[str, Any]]:
        if training is None:
            return []
        values: dict[int, dict[str, Any]] = {}
        for reference in training.wandb_runs:
            rows = self._wandb_index(reference.project_path).evaluations(
                run_name,
                directory,
                run_id=reference.run_id,
                metric=reference.validation_metric,
                artifact_step_offset=reference.rollout_artifact_offset,
            )
            values.update((int(row["optimizer_step"]), row) for row in rows)
        return [values[step] for step in sorted(values)]

    def validation_status(self, training_id: str) -> dict[str, Any]:
        training = self.training(training_id)
        if training is None or not training.wandb_runs:
            return {"state": "unconfigured", "fetched_at": None, "error": None}
        statuses = [
            {"wandb_run_id": reference.run_id, **self._wandb_index(reference.project_path).status(reference.run_id)}
            for reference in training.wandb_runs
        ]
        failed = [status for status in statuses if status["state"] == "error"]
        errors = [f"{status['wandb_run_id']}: {status.get('error') or 'unknown error'}" for status in failed]
        return {
            **statuses[-1],
            "state": "error" if failed else statuses[-1]["state"],
            "error": "; ".join(errors) if errors else statuses[-1].get("error"),
            "runs": statuses,
        }

    def training_metrics(self, training_id: str) -> dict[str, Any]:
        training = self.training(training_id)
        if training is None:
            return {"series": {}}
        merged: dict[str, dict[str, Any]] = {}
        for reference in training.wandb_runs:
            payload = self._wandb_index(reference.project_path).training_metrics(reference.run_id)
            for key, value in payload["series"].items():
                current = merged.setdefault(key, {"metric": None, "points": []})
                points = {int(point["optimizer_step"]): dict(point) for point in current["points"]}
                points.update((int(point["optimizer_step"]), dict(point)) for point in value["points"])
                current["metric"] = value["metric"] or current["metric"]
                current["points"] = [points[step] for step in sorted(points)]
        return {"series": merged}

    def training_progress(self, training_id: str) -> dict[str, Any]:
        training = self.training(training_id)
        if training is None or not training.wandb_runs:
            return {"optimizer_step": None, "run_state": None}
        progress = [
            {"wandb_run_id": reference.run_id, **self._wandb_index(reference.project_path).progress(reference.run_id)}
            for reference in training.wandb_runs
        ]
        steps = [item["optimizer_step"] for item in progress if item["optimizer_step"] is not None]
        return {**progress[-1], "optimizer_step": max(steps) if steps else None, "runs": progress}

    def _training_for(self, run_name: str, run_id: str | None) -> TrainingDefinition | None:
        with self._lock:
            if run_id:
                return next(
                    (
                        training
                        for training in self._trainings.values()
                        if any(reference.run_id == run_id for reference in training.wandb_runs)
                    ),
                    None,
                )
            item = self._attempts.get(run_name)
            return item[0] if item else None

    def _wandb_index(self, path: str) -> WandbEvalIndex:
        with self._lock:
            index = self._wandb.get(path)
            if index is None:
                index = WandbEvalIndex(path)
                self._wandb[path] = index
            return index
