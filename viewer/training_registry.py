from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import yaml

ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
GITHUB_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
DEFAULT_GIT_REPOSITORY = "allenai/open-instruct"


class TrainingRegistryError(ValueError):
    pass


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingRegistryError(f"{context} must be a mapping")
    return value


def _string(value: Any, context: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TrainingRegistryError(f"{context} must be a non-empty string")
    return value.strip()


def _resolve_path(repo_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _git_repository_url(value: str, context: str) -> str:
    repository = value.removesuffix(".git").rstrip("/")
    if repository.startswith("https://github.com/"):
        return repository
    if GITHUB_REPOSITORY_PATTERN.fullmatch(repository):
        return f"https://github.com/{repository}"
    raise TrainingRegistryError(f"{context} must be an owner/repository slug or a GitHub HTTPS URL")


@dataclasses.dataclass(frozen=True)
class WandbReference:
    entity: str
    project: str
    run_id: str
    validation_metric: str = "eval/objective/verifiable_correct_rate"
    rollout_artifact_offset: int = -1

    @property
    def project_path(self) -> str:
        return f"{self.entity}/{self.project}"

    @property
    def url(self) -> str:
        return f"https://wandb.ai/{self.project_path}/runs/{self.run_id}"

    def public(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "project": self.project,
            "run_id": self.run_id,
            "url": self.url,
            "validation_metric": self.validation_metric,
            "rollout_artifact_offset": self.rollout_artifact_offset,
        }


@dataclasses.dataclass(frozen=True)
class RolloutBinding:
    path: Path
    attempts: tuple[str, ...]
    configured_only: bool = False

    def public(self, repo_root: Path) -> dict[str, Any]:
        try:
            display_path = str(self.path.relative_to(repo_root))
        except ValueError:
            display_path = str(self.path)
        present_attempts = [
            attempt for attempt in self.attempts if (self.path / f"{attempt}_metadata.jsonl").is_file()
        ]
        return {
            "path": str(self.path),
            "display_path": display_path,
            "attempts": list(self.attempts),
            "present_attempts": present_attempts,
            "exists": self.path.is_dir(),
            "configured_only": self.configured_only,
        }


@dataclasses.dataclass(frozen=True)
class LaunchReference:
    id: str
    relation: str
    script: str | None
    historical_script: str | None
    git_commit: str | None
    git_repository: str
    beaker_experiment: str | None
    image: str | None
    wandb: WandbReference | None
    rollouts: tuple[RolloutBinding, ...]
    note: str | None = None
    checkpoint_state_dir: Path | None = None

    def _checkpoint_state_public(self) -> dict[str, Any] | None:
        """Live status of the DeepSpeed resume-state directory, if registered.

        These are the opaque <timestamp>_<pid> directories under
        deletable_checkpoint_states; the `latest` file names the resumable
        global step and each global_step<N>/ subdirectory is a resume point.
        """
        if self.checkpoint_state_dir is None:
            return None
        directory = self.checkpoint_state_dir
        exists = directory.is_dir()
        latest = None
        steps: list[int] = []
        if exists:
            try:
                latest = (directory / "latest").read_text(encoding="utf-8").strip() or None
            except OSError:
                latest = None
            try:
                steps = sorted(
                    int(entry.name[len("global_step") :])
                    for entry in directory.iterdir()
                    if entry.name.startswith("global_step") and entry.name[len("global_step") :].isdigit()
                )
            except OSError:
                steps = []
        return {"path": str(directory), "exists": exists, "latest": latest, "resumable_steps": steps}

    def public(self, repo_root: Path) -> dict[str, Any]:
        script_path = _resolve_path(repo_root, self.script)
        repository_url = _git_repository_url(self.git_repository, f"{self.id}.git_repository")
        return {
            "id": self.id,
            "relation": self.relation,
            "script": self.script,
            "script_path": str(script_path) if script_path else None,
            "script_exists": bool(script_path and script_path.is_file()),
            "historical_script": self.historical_script,
            "git_commit": self.git_commit,
            "git_repository": self.git_repository,
            "git_repository_url": repository_url,
            "git_url": f"{repository_url}/commit/{self.git_commit}" if self.git_commit else None,
            "beaker_experiment": self.beaker_experiment,
            "beaker_url": f"https://beaker.org/ex/{self.beaker_experiment}" if self.beaker_experiment else None,
            "image": self.image,
            "image_url": f"https://beaker.org/im/{self.image}" if self.image else None,
            "wandb": self.wandb.public() if self.wandb else None,
            "rollouts": [rollout.public(repo_root) for rollout in self.rollouts],
            "note": self.note,
            "checkpoint_state": self._checkpoint_state_public(),
        }


@dataclasses.dataclass(frozen=True)
class CheckpointReference:
    step: int
    path: Path

    def public(self) -> dict[str, Any]:
        # Registry checkpoint paths always name the checkpoint directory.  Do
        # not infer file-vs-directory from current existence: a deleted
        # checkpoint should remain visibly missing, rather than making its
        # existing parent directory look like a valid artifact.
        directory = self.path
        config = directory / "config.json"
        return {"step": self.step, "path": str(directory), "exists": directory.is_dir(), "complete": config.is_file()}


@dataclasses.dataclass(frozen=True)
class InferenceArtifact:
    path: Path
    schema: str = "open_instruct_inference_v1"
    judged_response_index: int = -1

    def public(self, repo_root: Path) -> dict[str, Any]:
        try:
            display_path = str(self.path.relative_to(repo_root))
        except ValueError:
            display_path = str(self.path)
        evaluation_dir = self.path / "eval"
        return {
            "path": str(self.path),
            "display_path": display_path,
            "schema": self.schema,
            "judged_response_index": self.judged_response_index,
            "exists": self.path.is_dir(),
            "complete": (
                (evaluation_dir / "evaluation_summary.json").is_file()
                and (evaluation_dir / "evaluation_results.jsonl").is_file()
                and (evaluation_dir / "judge_results").is_dir()
            ),
        }


@dataclasses.dataclass(frozen=True)
class BestEvaluation:
    benchmark: str
    step: int
    correct: int
    total: int
    checkpoint: CheckpointReference | None = None
    inference_artifact: InferenceArtifact | None = None
    beaker_experiment: str | None = None

    @property
    def id(self) -> str:
        benchmark = re.sub(r"[^a-z0-9]+", "-", self.benchmark.casefold()).strip("-")
        return f"{benchmark}-step-{self.step}"

    @property
    def score(self) -> float:
        return self.correct / self.total

    def public(self, repo_root: Path) -> dict[str, Any]:
        return {
            "id": self.id,
            "benchmark": self.benchmark,
            "step": self.step,
            "correct": self.correct,
            "total": self.total,
            "score": self.score,
            "checkpoint": self.checkpoint.public() if self.checkpoint else None,
            "inference_artifact": (self.inference_artifact.public(repo_root) if self.inference_artifact else None),
            "beaker_experiment": self.beaker_experiment,
            "beaker_url": f"https://beaker.org/ex/{self.beaker_experiment}" if self.beaker_experiment else None,
        }


@dataclasses.dataclass(frozen=True)
class TrainingDefinition:
    id: str
    title: str
    classification: str
    visibility: str
    tags: dict[str, str]
    wandb: WandbReference | None
    launches: tuple[LaunchReference, ...]
    furthest_step: int | None
    best_evaluation: BestEvaluation | None
    evaluations: tuple[BestEvaluation, ...]
    latest_checkpoint: CheckpointReference | None
    note: str | None = None
    checkpoints: tuple[CheckpointReference, ...] = ()

    @property
    def rollout_attempts(self) -> tuple[str, ...]:
        return tuple(
            attempt for launch in self.launches for rollout in launch.rollouts for attempt in rollout.attempts
        )

    @property
    def wandb_runs(self) -> tuple[WandbReference, ...]:
        references = ([self.wandb] if self.wandb else []) + [
            launch.wandb for launch in self.launches if launch.wandb is not None
        ]
        unique: dict[tuple[str, str, str], WandbReference] = {}
        for reference in references:
            key = (reference.entity, reference.project, reference.run_id)
            unique.pop(key, None)
            unique[key] = reference
        return tuple(unique.values())

    @property
    def active_wandb(self) -> WandbReference | None:
        runs = self.wandb_runs
        return runs[-1] if runs else None

    def public(self, repo_root: Path) -> dict[str, Any]:
        active_wandb = self.active_wandb
        return {
            "id": self.id,
            "title": self.title,
            "classification": self.classification,
            "visibility": self.visibility,
            "tags": self.tags,
            "wandb": active_wandb.public() if active_wandb else None,
            "wandb_runs": [reference.public() for reference in self.wandb_runs],
            "launches": [launch.public(repo_root) for launch in self.launches],
            "furthest_step": self.furthest_step,
            "best_evaluation": self.best_evaluation.public(repo_root) if self.best_evaluation else None,
            "evaluations": [evaluation.public(repo_root) for evaluation in self.evaluations],
            "latest_checkpoint": self.latest_checkpoint.public() if self.latest_checkpoint else None,
            "checkpoints": [checkpoint.public() for checkpoint in self.checkpoints],
            "note": self.note,
        }


class TrainingRegistry:
    def __init__(self, path: str | Path) -> None:
        requested_path = Path(path).expanduser().resolve()
        self.root = requested_path if requested_path.is_dir() else requested_path.parent
        self.path = self.root / "config.yaml" if requested_path.is_dir() else requested_path
        self.trainings_dir = self.root / "trainings"
        # Backward-compatible default for standalone registries. The checked-in
        # nested registry sets repo_root explicitly in config.yaml.
        self.repo_root = self.root.parent.resolve()
        self.defaults: dict[str, Any] = {}
        self.trainings: tuple[TrainingDefinition, ...] = ()
        self.refresh()

    def refresh(self) -> None:
        if not self.path.is_file():
            raise TrainingRegistryError(f"Training registry config does not exist: {self.path}")
        config = self._load_yaml(self.path, "registry config")
        if config.get("schema_version") != 1:
            raise TrainingRegistryError(f"{self.path}: schema_version must be 1")
        if config.get("kind") != "training_registry":
            raise TrainingRegistryError(f"{self.path}: kind must be 'training_registry'")
        repo_root_value = config.get("repo_root")
        if repo_root_value is not None:
            repo_root = _string(repo_root_value, f"{self.path}: repo_root")
            assert repo_root is not None
            candidate = Path(repo_root).expanduser()
            self.repo_root = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
        else:
            self.repo_root = self.root.parent.resolve()
        defaults = _mapping(config.get("defaults") or {}, f"{self.path}: defaults")
        if not self.trainings_dir.is_dir():
            raise TrainingRegistryError(f"Training registry directory does not exist: {self.trainings_dir}")

        parsed: list[tuple[Path, TrainingDefinition]] = []
        for source in sorted(self.trainings_dir.glob("*.yaml"), key=lambda item: item.name):
            document = self._load_yaml(source, "training entry")
            if document.get("schema_version") != 1:
                raise TrainingRegistryError(f"{source}: schema_version must be 1")
            if document.get("kind") != "training":
                raise TrainingRegistryError(f"{source}: kind must be 'training'")
            row = {key: value for key, value in document.items() if key not in {"schema_version", "kind"}}
            training = self._parse_training(row, defaults, str(source))
            if source.stem != training.id:
                raise TrainingRegistryError(f"{source}: filename must match training id {training.id!r}")
            parsed.append((source, training))

        seen_ids: dict[str, Path] = {}
        seen_wandb_ids: dict[str, tuple[str, Path]] = {}
        seen_attempts: dict[str, str] = {}
        seen_inference_artifacts: dict[Path, tuple[str, str]] = {}
        for source, training in parsed:
            owner_file = seen_ids.setdefault(training.id, source)
            if owner_file != source:
                raise TrainingRegistryError(f"duplicate training id {training.id!r} in {owner_file} and {source}")
            for reference in training.wandb_runs:
                owner = seen_wandb_ids.setdefault(reference.run_id, (training.id, source))
                if owner != (training.id, source):
                    raise TrainingRegistryError(
                        f"W&B run {reference.run_id!r} belongs to both "
                        f"{owner[0]!r} ({owner[1]}) and {training.id!r} ({source})"
                    )
            for attempt in training.rollout_attempts:
                owner = seen_attempts.setdefault(attempt, training.id)
                if owner != training.id:
                    raise TrainingRegistryError(
                        f"rollout attempt {attempt!r} belongs to both {owner!r} and {training.id!r}"
                    )
            for evaluation in training.evaluations:
                artifact = evaluation.inference_artifact
                if artifact is None:
                    continue
                artifact_path = artifact.path.resolve()
                owner = seen_inference_artifacts.setdefault(artifact_path, (training.id, evaluation.id))
                if owner != (training.id, evaluation.id):
                    raise TrainingRegistryError(
                        f"inference artifact {artifact_path} belongs to both "
                        f"{owner[0]!r}/{owner[1]!r} and {training.id!r}/{evaluation.id!r}"
                    )
        trainings = tuple(training for _, training in parsed)
        self.defaults = defaults
        self.trainings = trainings

    @staticmethod
    def _load_yaml(path: Path, label: str) -> dict[str, Any]:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise TrainingRegistryError(f"Could not read {label} {path}: {error}") from error
        return _mapping(payload, f"{label} {path}")

    def get(self, training_id: str) -> TrainingDefinition:
        try:
            return next(training for training in self.trainings if training.id == training_id)
        except StopIteration as error:
            raise TrainingRegistryError(f"Unknown training: {training_id}") from error

    def public(self) -> list[dict[str, Any]]:
        return [training.public(self.repo_root) for training in self.trainings]

    def _parse_training(self, value: Any, defaults: dict[str, Any], context: str) -> TrainingDefinition:
        row = _mapping(value, context)
        training_id = _string(row.get("id"), f"{context}.id")
        assert training_id is not None
        if not ID_PATTERN.fullmatch(training_id):
            raise TrainingRegistryError(f"invalid training id: {training_id!r}")
        wandb = self._parse_wandb(row.get("wandb"), defaults, training_id)
        launches_value = row.get("launches") or []
        if not isinstance(launches_value, list):
            raise TrainingRegistryError(f"{training_id}.launches must be a list")
        launches = tuple(
            self._parse_launch(item, training_id, i, defaults=defaults, fallback_wandb=wandb)
            for i, item in enumerate(launches_value)
        )
        tags = _mapping(row.get("tags") or {}, f"{training_id}.tags")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in tags.items()):
            raise TrainingRegistryError(f"{training_id}.tags values must be strings")
        artifacts = _mapping(row.get("artifacts") or {}, f"{training_id}.artifacts")
        best = self._parse_best_evaluation(artifacts.get("best_evaluation"), training_id)
        evaluation_rows = artifacts.get("evaluations") or []
        if not isinstance(evaluation_rows, list):
            raise TrainingRegistryError(f"{training_id}.evaluations must be a list")
        parsed_evaluations = [
            self._parse_evaluation(item, f"{training_id}.evaluations[{index}]")
            for index, item in enumerate(evaluation_rows)
        ]
        evaluations_by_step: dict[tuple[str, int], BestEvaluation] = {}
        for evaluation in parsed_evaluations:
            key = (evaluation.benchmark.casefold(), evaluation.step)
            if key in evaluations_by_step:
                raise TrainingRegistryError(
                    f"{training_id}.evaluations contains duplicate benchmark/step {evaluation.benchmark!r}/{evaluation.step}"
                )
            evaluations_by_step[key] = evaluation
        if best is not None:
            # The legacy best_evaluation row remains authoritative when the
            # same result is also included in the new history list, since it
            # commonly carries the retained checkpoint path.
            key = (best.benchmark.casefold(), best.step)
            previous = evaluations_by_step.get(key)
            if previous is not None:
                best = dataclasses.replace(
                    best,
                    checkpoint=best.checkpoint or previous.checkpoint,
                    inference_artifact=best.inference_artifact or previous.inference_artifact,
                )
            evaluations_by_step[key] = best
        evaluations = tuple(
            sorted(evaluations_by_step.values(), key=lambda item: (item.benchmark.casefold(), item.step))
        )
        if best is None and evaluations:
            best = max(evaluations, key=lambda item: (item.score, item.step))
        latest = self._parse_checkpoint(artifacts.get("latest_checkpoint"), f"{training_id}.latest_checkpoint")
        checkpoint_rows = artifacts.get("checkpoints") or []
        if not isinstance(checkpoint_rows, list):
            raise TrainingRegistryError(f"{training_id}.checkpoints must be a list")
        checkpoints = tuple(
            sorted(
                (
                    self._parse_checkpoint(item, f"{training_id}.checkpoints[{index}]")
                    for index, item in enumerate(checkpoint_rows)
                ),
                key=lambda checkpoint: checkpoint.step,
            )
        )
        furthest = artifacts.get("furthest_step")
        if furthest is not None and (not isinstance(furthest, int) or furthest < 0):
            raise TrainingRegistryError(f"{training_id}.furthest_step must be a non-negative integer")
        return TrainingDefinition(
            id=training_id,
            title=_string(row.get("title"), f"{training_id}.title") or training_id,
            classification=_string(row.get("classification", "substantive"), f"{training_id}.classification")
            or "substantive",
            visibility=_string(row.get("visibility", "default"), f"{training_id}.visibility") or "default",
            tags=dict(tags),
            wandb=wandb,
            launches=launches,
            furthest_step=furthest,
            best_evaluation=best,
            evaluations=evaluations,
            latest_checkpoint=latest,
            note=_string(row.get("note"), f"{training_id}.note", required=False),
            checkpoints=checkpoints,
        )

    def _parse_wandb(
        self, value: Any, defaults: dict[str, Any], context: str, fallback: WandbReference | None = None
    ) -> WandbReference | None:
        if value is None:
            return None
        row = _mapping(value, f"{context}.wandb")
        default_wandb = _mapping(defaults.get("wandb") or {}, "defaults.wandb")
        return WandbReference(
            entity=_string(
                row.get("entity", fallback.entity if fallback else default_wandb.get("entity")),
                f"{context}.wandb.entity",
            )
            or "",
            project=_string(
                row.get("project", fallback.project if fallback else default_wandb.get("project")),
                f"{context}.wandb.project",
            )
            or "",
            run_id=_string(row.get("run_id"), f"{context}.wandb.run_id") or "",
            validation_metric=_string(
                row.get(
                    "validation_metric",
                    fallback.validation_metric
                    if fallback
                    else default_wandb.get("validation_metric", WandbReference.validation_metric),
                ),
                f"{context}.wandb.validation_metric",
            )
            or WandbReference.validation_metric,
            rollout_artifact_offset=int(
                row.get("rollout_artifact_offset", fallback.rollout_artifact_offset if fallback else -1)
            ),
        )

    def _parse_launch(
        self,
        value: Any,
        training_id: str,
        index: int,
        *,
        defaults: dict[str, Any],
        fallback_wandb: WandbReference | None,
    ) -> LaunchReference:
        row = _mapping(value, f"{training_id}.launches[{index}]")
        launch_id = _string(row.get("id", f"launch-{index + 1}"), f"{training_id}.launches[{index}].id")
        git_repository = (
            _string(row.get("git_repository", DEFAULT_GIT_REPOSITORY), f"{training_id}.{launch_id}.git_repository")
            or DEFAULT_GIT_REPOSITORY
        )
        _git_repository_url(git_repository, f"{training_id}.{launch_id}.git_repository")
        rollout_rows = row.get("rollouts") or []
        if not isinstance(rollout_rows, list):
            raise TrainingRegistryError(f"{training_id}.{launch_id}.rollouts must be a list")
        rollouts = []
        for rollout_index, rollout_value in enumerate(rollout_rows):
            rollout = _mapping(rollout_value, f"{training_id}.{launch_id}.rollouts[{rollout_index}]")
            raw_attempts = rollout.get("attempts") or []
            if not isinstance(raw_attempts, list) or any(not isinstance(item, str) for item in raw_attempts):
                raise TrainingRegistryError(
                    f"{training_id}.{launch_id}.rollouts[{rollout_index}].attempts must be strings"
                )
            path_value = _string(rollout.get("path"), f"{training_id}.{launch_id}.rollouts[{rollout_index}].path")
            rollouts.append(
                RolloutBinding(
                    path=_resolve_path(self.repo_root, path_value) or self.repo_root,
                    attempts=tuple(raw_attempts),
                    configured_only=bool(rollout.get("configured_only", False)),
                )
            )
        return LaunchReference(
            id=launch_id or f"launch-{index + 1}",
            relation=_string(row.get("relation", "initial"), f"{training_id}.{launch_id}.relation") or "initial",
            script=_string(row.get("script"), f"{training_id}.{launch_id}.script", required=False),
            historical_script=_string(
                row.get("historical_script"), f"{training_id}.{launch_id}.historical_script", required=False
            ),
            git_commit=_string(row.get("git_commit"), f"{training_id}.{launch_id}.git_commit", required=False),
            git_repository=git_repository,
            beaker_experiment=_string(
                row.get("beaker_experiment"), f"{training_id}.{launch_id}.beaker_experiment", required=False
            ),
            image=_string(row.get("image"), f"{training_id}.{launch_id}.image", required=False),
            wandb=(
                fallback_wandb
                if row.get("wandb") is None
                else self._parse_wandb(
                    row.get("wandb"), defaults, f"{training_id}.{launch_id}", fallback=fallback_wandb
                )
            ),
            rollouts=tuple(rollouts),
            checkpoint_state_dir=_resolve_path(
                self.repo_root,
                _string(
                    row.get("checkpoint_state_dir"), f"{training_id}.{launch_id}.checkpoint_state_dir", required=False
                ),
            ),
            note=_string(row.get("note"), f"{training_id}.{launch_id}.note", required=False),
        )

    def _parse_checkpoint(self, value: Any, context: str) -> CheckpointReference | None:
        if value is None:
            return None
        row = _mapping(value, context)
        step = row.get("step")
        if not isinstance(step, int) or step < 0:
            raise TrainingRegistryError(f"{context}.step must be a non-negative integer")
        raw_path = _string(row.get("path"), f"{context}.path")
        return CheckpointReference(step=step, path=_resolve_path(self.repo_root, raw_path) or self.repo_root)

    def _parse_best_evaluation(self, value: Any, training_id: str) -> BestEvaluation | None:
        if value is None:
            return None
        return self._parse_evaluation(value, f"{training_id}.best_evaluation")

    def _parse_evaluation(self, value: Any, context: str) -> BestEvaluation:
        row = _mapping(value, context)
        step, correct, total = row.get("step"), row.get("correct"), row.get("total")
        if not isinstance(step, int) or not isinstance(correct, int) or not isinstance(total, int):
            raise TrainingRegistryError(f"{context} requires valid integer step/correct/total")
        if not 0 <= correct <= total or total <= 0:
            raise TrainingRegistryError(f"{context} requires valid integer step/correct/total")
        checkpoint = self._parse_checkpoint(row.get("checkpoint"), f"{context}.checkpoint")
        inference_artifact = self._parse_inference_artifact(
            row.get("inference_artifact"), f"{context}.inference_artifact"
        )
        beaker_experiment = _string(row.get("beaker_experiment"), f"{context}.beaker_experiment", required=False)
        if beaker_experiment:
            # Accept a full URL in the registry but store the bare experiment id.
            beaker_experiment = beaker_experiment.rstrip("/").rsplit("/", 1)[-1]
        return BestEvaluation(
            benchmark=_string(row.get("benchmark"), f"{context}.benchmark") or "",
            step=step,
            correct=correct,
            total=total,
            checkpoint=checkpoint,
            inference_artifact=inference_artifact,
            beaker_experiment=beaker_experiment or None,
        )

    def _parse_inference_artifact(self, value: Any, context: str) -> InferenceArtifact | None:
        if value is None:
            return None
        row = _mapping(value, context)
        raw_path = _string(row.get("path"), f"{context}.path")
        schema = _string(row.get("schema", "open_instruct_inference_v1"), f"{context}.schema")
        if schema != "open_instruct_inference_v1":
            raise TrainingRegistryError(f"{context}.schema must be 'open_instruct_inference_v1'")
        judged_response_index = row.get("judged_response_index", -1)
        if not isinstance(judged_response_index, int):
            raise TrainingRegistryError(f"{context}.judged_response_index must be an integer")
        return InferenceArtifact(
            path=_resolve_path(self.repo_root, raw_path) or self.repo_root,
            schema=schema,
            judged_response_index=judged_response_index,
        )
