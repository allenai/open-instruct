from __future__ import annotations

import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote

from viewer.evaluation_store import EvaluationStore
from viewer.registry_index import RegistryIndex
from viewer.rollout_store import RolloutStore
from viewer.training_registry import TrainingDefinition, TrainingRegistry, TrainingRegistryError


class ExperimentService:
    """Join the checked-in training catalog with live filesystem and W&B state."""

    def __init__(self, registry: TrainingRegistry, store: RolloutStore, registry_index: RegistryIndex) -> None:
        self.registry = registry
        self.store = store
        self.registry_index = registry_index
        self.evaluation_store = EvaluationStore(registry)
        self._lock = threading.RLock()
        self._validations: dict[str, dict[str, Any]] = {}
        self._refresh_thread: threading.Thread | None = None
        self._refresh_state: dict[str, Any] = {
            "state": "pending",
            "started_at": None,
            "finished_at": None,
            "completed": 0,
            "total": 0,
        }

    def refresh_catalog(self) -> None:
        self.registry_index.refresh_registry()
        self.registry_index.invalidate()
        self.store.refresh()
        self.evaluation_store.refresh()

    def start_validation_refresh(self, *, force: bool = True) -> bool:
        """Re-read W&B in the background without blocking viewer startup."""
        with self._lock:
            if self._refresh_thread is not None and self._refresh_thread.is_alive():
                return False
            targets = [training for training in self.registry.trainings if training.wandb_runs]
            now = datetime.datetime.now(datetime.UTC).isoformat()
            self._refresh_state = {
                "state": "refreshing",
                "started_at": now,
                "finished_at": None,
                "completed": 0,
                "total": len(targets),
            }
            # Keep the last successful data visible, but make each target's
            # in-flight state explicit. The detail page polls this status. If
            # it remains "fresh" from the previous fetch, the page can stop
            # polling before this training is reached by the background loop,
            # leaving summary cards stale while separately fetched charts are
            # already current.
            for training in targets:
                previous = self._validations.get(training.id)
                if previous is None:
                    self._validations[training.id] = self._build_validation_payload(
                        [], {"state": "refreshing", "fetched_at": None, "error": None}
                    )
                else:
                    self._validations[training.id] = {
                        **previous,
                        "status": {**previous["status"], "state": "refreshing", "error": None},
                    }
            thread = threading.Thread(
                target=self._refresh_validations, args=(targets, force), name="viewer-wandb-refresh", daemon=True
            )
            self._refresh_thread = thread
            thread.start()
            return True

    def _refresh_validations(self, targets: list[TrainingDefinition], force: bool) -> None:
        if force:
            for training in targets:
                self.registry_index.invalidate(training.id, include_training_metrics=False)

        def refresh_one(training: TrainingDefinition) -> None:
            previous = self._validation_snapshot(training.id)
            evaluations = self.registry_index.validation_evaluations(training.id)
            progress = self.registry_index.training_progress(training.id)
            status = self.registry_index.validation_status(training.id)
            if status["state"] == "error" and previous.get("evaluations"):
                payload = dict(previous)
                payload["status"] = {**status, "state": "stale"}
                payload["progress"] = progress
            else:
                payload = self._build_validation_payload(evaluations, status, progress)
            with self._lock:
                self._validations[training.id] = payload
                self._refresh_state["completed"] += 1

        # Each training's refresh is dominated by W&B round-trips; a serial
        # walk left later catalog rows unpopulated for minutes.
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="validation-refresh") as pool:
            list(pool.map(refresh_one, targets))
        with self._lock:
            self._refresh_state["state"] = "complete"
            self._refresh_state["finished_at"] = datetime.datetime.now(datetime.UTC).isoformat()

    @staticmethod
    def _build_validation_payload(
        evaluations: list[dict[str, Any]], status: dict[str, Any], progress: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        latest = evaluations[-1] if evaluations else None
        best = max(evaluations, key=lambda row: (row["score"], row["optimizer_step"])) if evaluations else None
        return {"evaluations": evaluations, "latest": latest, "best": best, "status": status, "progress": progress}

    def _validation_snapshot(self, training_id: str) -> dict[str, Any]:
        with self._lock:
            value = self._validations.get(training_id)
            if value is not None:
                return {
                    **value,
                    "evaluations": [dict(item) for item in value["evaluations"]],
                    "latest": dict(value["latest"]) if value["latest"] else None,
                    "best": dict(value["best"]) if value["best"] else None,
                    "status": dict(value["status"]),
                    "progress": dict(value["progress"]) if value.get("progress") else None,
                }
        status = self.registry_index.validation_status(training_id)
        return self._build_validation_payload([], status)

    def list_trainings(self) -> dict[str, Any]:
        public = {row["id"]: row for row in self.registry.public()}
        rollout_runs = {
            row["registry_id"]: row for row in self.store.meta()["runs"] if row.get("registry_id") is not None
        }
        trainings = [
            self._join_training(public[training.id], rollout_runs.get(training.id))
            for training in self.registry.trainings
        ]
        with self._lock:
            refresh_state = dict(self._refresh_state)
        counts: dict[str, int] = {}
        for row in trainings:
            counts[row["classification"]] = counts.get(row["classification"], 0) + 1
        return {
            "trainings": trainings,
            "summary": {
                "total": len(trainings),
                "classifications": counts,
                "inspectable_rollouts": sum(row["live"]["rollouts"]["available"] for row in trainings),
                "validation_refresh": refresh_state,
            },
            "refreshed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }

    def get_training(self, training_id: str) -> dict[str, Any]:
        rows = self.list_trainings()["trainings"]
        try:
            return next(row for row in rows if row["id"] == training_id)
        except StopIteration as error:
            raise TrainingRegistryError(f"Unknown training: {training_id}") from error

    def get_training_metrics(self, training_id: str) -> dict[str, Any]:
        """Load detail-only W&B curves without inflating the catalog payload."""
        self.registry.get(training_id)
        payload = self.registry_index.training_metrics(training_id)
        return {"training_id": training_id, **payload, "status": self.registry_index.validation_status(training_id)}

    def get_evaluation_records(
        self, *, training_id: str, evaluation_id: str, outcome: str, search: str, sort: str, page: int, page_size: int
    ) -> dict[str, Any]:
        return self.evaluation_store.query(
            training_id=training_id,
            evaluation_id=evaluation_id,
            outcome=outcome,
            search=search,
            sort=sort,
            page=page,
            page_size=page_size,
        )

    def get_evaluation_record(
        self, *, training_id: str, evaluation_id: str, query_id: str, response_index: int | None
    ) -> dict[str, Any]:
        return self.evaluation_store.detail(
            training_id=training_id, evaluation_id=evaluation_id, query_id=query_id, response_index=response_index
        )

    def search_evaluation_record(
        self, *, training_id: str, evaluation_id: str, query_id: str, query: str, response_index: int | None
    ) -> dict[str, Any]:
        return self.evaluation_store.matches(
            training_id=training_id,
            evaluation_id=evaluation_id,
            query_id=query_id,
            query=query,
            response_index=response_index,
        )

    def get_evaluation_segment(
        self, *, training_id: str, evaluation_id: str, query_id: str, segment_index: int, response_index: int | None
    ) -> dict[str, Any]:
        return self.evaluation_store.segment(
            training_id=training_id,
            evaluation_id=evaluation_id,
            query_id=query_id,
            segment_index=segment_index,
            response_index=response_index,
        )

    def _join_training(self, training: dict[str, Any], rollout_run: dict[str, Any] | None) -> dict[str, Any]:
        local = self._add_local_urls(training)
        validation = self._validation_snapshot(training["id"])
        live_step = (validation.get("progress") or {}).get("optimizer_step")
        if live_step is not None:
            local["furthest_step"] = max(local.get("furthest_step") or 0, int(live_step))
        rollout = {
            "available": rollout_run is not None,
            "logical_run": rollout_run["name"] if rollout_run else None,
            "attempts": list(rollout_run["attempts"]) if rollout_run else [],
            "accepted_files": rollout_run["accepted_files"] if rollout_run else 0,
            "filtered_files": rollout_run["filtered_files"] if rollout_run else 0,
            "first_step": rollout_run["first_step"] if rollout_run else None,
            "last_step": rollout_run["last_step"] if rollout_run else None,
            "resolved": rollout_run["resolved"] if rollout_run else False,
            "updated": rollout_run["updated"] if rollout_run else None,
            "attempt_metadata": list(rollout_run.get("attempt_metadata", [])) if rollout_run else [],
        }
        return {**local, "live": {"validation": validation, "rollouts": rollout}}

    def _add_local_urls(self, training: dict[str, Any]) -> dict[str, Any]:
        for launch in training["launches"]:
            if launch["script_path"]:
                launch["script_url"] = self._path_url(launch["script_path"])
            for rollout in launch["rollouts"]:
                rollout["path_url"] = self._path_url(rollout["path"])
        for checkpoint_key in ("latest_checkpoint",):
            checkpoint = training.get(checkpoint_key)
            if checkpoint:
                checkpoint["path_url"] = self._path_url(checkpoint["path"])
        best = training.get("best_evaluation")
        if best and best.get("checkpoint"):
            best["checkpoint"]["path_url"] = self._path_url(best["checkpoint"]["path"])
        for evaluation in training.get("evaluations", []):
            artifact = evaluation.get("inference_artifact")
            if artifact:
                artifact["path_url"] = self._path_url(artifact["path"])
        return training

    @staticmethod
    def _path_url(path: str) -> str:
        return f"/api/path?path={quote(path, safe='')}"

    def path_info(self, raw_path: str) -> dict[str, Any]:
        requested = Path(raw_path).expanduser().resolve()
        allowed = self._registered_paths()
        if requested not in allowed:
            raise TrainingRegistryError("Path is not registered for a training experiment")
        payload: dict[str, Any] = {"path": str(requested), "exists": requested.exists(), "kind": "missing"}
        if requested.is_dir():
            payload["kind"] = "directory"
            try:
                children = sorted(requested.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
            except OSError as error:
                payload["error"] = str(error)
                return payload
            payload["entries"] = [
                {
                    "name": child.name,
                    "kind": "directory" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
                for child in children[:200]
            ]
            payload["truncated"] = len(children) > 200
        elif requested.is_file():
            payload["kind"] = "file"
            payload["size"] = requested.stat().st_size
            if requested.stat().st_size <= 262_144:
                try:
                    payload["content"] = requested.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    payload["content"] = None
        return payload

    def _registered_paths(self) -> set[Path]:
        paths: set[Path] = set()
        for training in self.registry.trainings:
            for launch in training.launches:
                script = (self.registry.repo_root / launch.script).resolve() if launch.script else None
                if script:
                    paths.add(script)
                paths.update(rollout.path.resolve() for rollout in launch.rollouts)
            if training.latest_checkpoint:
                paths.add(training.latest_checkpoint.path.resolve())
            if training.best_evaluation and training.best_evaluation.checkpoint:
                paths.add(training.best_evaluation.checkpoint.path.resolve())
            paths.update(
                evaluation.checkpoint.path.resolve()
                for evaluation in training.evaluations
                if evaluation.checkpoint is not None
            )
            paths.update(
                evaluation.inference_artifact.path.resolve()
                for evaluation in training.evaluations
                if evaluation.inference_artifact is not None
            )
        return paths
