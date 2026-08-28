"""Resolve rollout attempts into W&B lineages and load validation scores.

A restarted trainer gets a new timestamped ``run_name`` but can continue the
same W&B run.  Rollout files therefore describe process attempts, while W&B is
the source of truth for the logical training lineage and its validation history.

Everything here is best effort.  If W&B is unavailable, the rollout viewer
keeps attempts separate instead of risking an incorrect merge.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from numbers import Real
from pathlib import Path
from typing import Any

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

# Ordered by how directly they express answer correctness.
EVAL_METRIC_PREFERENCE = (
    "eval/objective/verifiable_correct_rate",
    "eval/objective/verifiable_reward",
    "eval/pass_at_1",
    "eval/scores",
)

# Compact scalar histories used by the training workspace.  Each series is
# resolved in priority order so older runs can still render when metric names
# evolved.  Raw per-token logprobs are intentionally excluded: loading those
# arrays would defeat the rollout viewer's lazy-I/O design.
TRAINING_METRIC_PREFERENCE = {
    "reward": ("scores", "val/avg_group_performance_post_filter"),
    "group_pass_rate_all": ("val/avg_group_performance_pre_filter",),
    "group_pass_rate_post_mask": ("val/avg_group_performance_post_filter",),
    "length": ("val/sequence_lengths",),
    "terminal_length": ("val/terminal_turn_lengths",),
    "token_capped": ("val/truncated_completion_fraction",),
    "tool_calls": ("tools/aggregate/avg_calls_per_rollout",),
    "search_failure_rate": ("tools/search/failure_rate",),
    "visit_failure_rate": ("tools/visit/failure_rate",),
    "format_incomplete_pre": ("format/incomplete_pre_filtering",),
    "format_terminal_pre": ("format/terminal_format_pre_filtering",),
    "format_trajectory_pre": ("format/trajectory_format_pre_filtering",),
    "format_incomplete_post": ("format/incomplete_post_filtering",),
    "format_terminal_post": ("format/terminal_format_post_filtering",),
    "format_trajectory_post": ("format/trajectory_format_post_filtering",),
    "logprob": ("debug/vllm_vs_local_logprob_diff_mean", "policy/entropy_avg"),
}
REJECTED_GROUP_METRICS = {
    "rejected": "batch/filtered_prompts",
    "all_zero": "batch/filtered_prompts_zero",
    "all_one": "batch/filtered_prompts_solved",
    "accepted": "batch/total_prompts",
}
REJECTED_GROUP_SERIES = ("rejected_group_rate", "rejected_all_zero_rate", "rejected_all_one_rate")
# W&B run states whose scalar history can no longer change; only these are
# served from the on-disk cache. A running/unknown run is always refetched.
TERMINAL_RUN_STATES = frozenset({"finished", "failed", "crashed", "killed"})

RUN_NAME = re.compile(r"^(?P<base>.+)__+(?P<seed>\d+)__+(?P<started>\d+)$")


def artifact_step(wandb_step: int) -> int:
    """Convert a W&B optimizer step to the zero-based rollout artifact step."""
    return int(wandb_step) - 1


def split_run_name(run_name: str) -> tuple[str, int | None]:
    """Return the stable experiment/seed prefix and process start timestamp."""
    match = RUN_NAME.match(run_name)
    if match is None:
        return run_name, None
    return f"{match.group('base')}__{match.group('seed')}", int(match.group("started"))


def _timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class WandbEvalIndex:
    """Resolve physical rollout attempts to W&B runs and validation results."""

    def __init__(
        self, path: str, overrides: dict[str, str] | None = None, cache_dir: str | Path | None = None
    ) -> None:
        self.path = path
        # Maps an attempt name or rollout directory name to a W&B run id.
        self.overrides = dict(overrides or {})
        self._lock = threading.RLock()
        # One Api (and its HTTP session) per thread: the fetch pool below runs
        # scan_history calls concurrently and requests sessions are not
        # guaranteed thread-safe.
        self._thread_state = threading.local()
        # Test/injection override: when set, every thread uses this client.
        self._api: Any = None
        self._fetch_pool = ThreadPoolExecutor(max_workers=12, thread_name_prefix="wandb-fetch")
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self._disk_entries: dict[str, dict[str, Any]] = {}
        self._catalog: list[dict[str, Any]] | None = None
        self._evaluations_cache: dict[tuple[str, str | None, int], list[dict[str, Any]]] = {}
        self._training_metrics_cache: dict[str, dict[str, Any]] = {}
        self._training_metrics_locks: dict[str, threading.Lock] = {}
        self._progress_cache: dict[str, dict[str, Any]] = {}
        self._validation_errors: dict[str, str] = {}
        self._validation_fetched_at: dict[str, str] = {}

    def _get_api(self) -> Any:
        if self._api is not None:
            return self._api
        api = getattr(self._thread_state, "api", None)
        if api is None:
            # Api construction reads settings/netrc, which races when several
            # fetch-pool threads initialize at once and can come up keyless.
            # Only construction is serialized; requests stay concurrent.
            with self._lock:
                api = import_module("wandb").Api()
            self._thread_state.api = api
        return api

    def _run_handle(self, run_id: str) -> Any:
        return self._get_api().run(f"{self.path}/{run_id}")

    # ------------------------------------------------------------------
    # On-disk cache. A finished/crashed/failed run's scalar history is
    # immutable, so it is stored once under <cache_dir>/<run_id>.json and
    # every later server start reads it back instead of re-crawling W&B.
    # ------------------------------------------------------------------

    def _disk_file(self, run_id: str) -> Path:
        return self._cache_dir / f"{run_id}.json"

    def _disk_entry(self, run_id: str) -> dict[str, Any]:
        if self._cache_dir is None:
            return {}
        with self._lock:
            cached = self._disk_entries.get(run_id)
        if cached is not None:
            return cached
        entry: dict[str, Any] = {}
        try:
            entry = json.loads(self._disk_file(run_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            entry = {}
        with self._lock:
            self._disk_entries[run_id] = entry
        return entry

    def _disk_get(self, run_id: str, section: str, key: str) -> Any:
        entry = self._disk_entry(run_id)
        if entry.get("run_state") not in TERMINAL_RUN_STATES:
            return None
        return (entry.get(section) or {}).get(key)

    def _disk_put(self, run_id: str, run_state: Any, section: str, key: str, payload: Any) -> None:
        """Persist one fetched section, but only once the run can no longer change."""
        if self._cache_dir is None or str(run_state) not in TERMINAL_RUN_STATES:
            return
        self._disk_entry(run_id)  # ensure any existing file is loaded before merging
        with self._lock:
            entry = dict(self._disk_entries.get(run_id) or {})
            section_map = dict(entry.get(section) or {})
            section_map[key] = payload
            entry[section] = section_map
            entry["run_state"] = str(run_state)
            entry["saved_at"] = datetime.datetime.now(datetime.UTC).isoformat()
            self._disk_entries[run_id] = entry
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                scratch = self._disk_file(run_id).with_suffix(".tmp")
                scratch.write_text(json.dumps(entry), encoding="utf-8")
                os.replace(scratch, self._disk_file(run_id))
            except OSError as error:
                logger.warning("W&B disk cache write failed for %s: %s", run_id, error)

    def _disk_drop(self, run_id: str | None = None) -> None:
        if self._cache_dir is None:
            return
        with self._lock:
            run_ids = [run_id] if run_id else list(self._disk_entries)
            if run_id is None and self._cache_dir.is_dir():
                run_ids = list({*run_ids, *(f.stem for f in self._cache_dir.glob("*.json"))})
            for target in run_ids:
                self._disk_entries.pop(target, None)
                try:
                    self._disk_file(target).unlink(missing_ok=True)
                except OSError:
                    pass

    def _get_catalog(self) -> list[dict[str, Any]]:
        """List the project once, retaining fields used for lineage matching."""
        with self._lock:
            if self._catalog is not None:
                return self._catalog
        catalog: list[dict[str, Any]] = []
        for run in self._get_api().runs(self.path):
            config = run.config or {}
            run_name = str(config.get("run_name") or "")
            display_name = run.name or ""
            stable_name, _ = split_run_name(run_name or display_name)
            catalog.append(
                {
                    "id": run.id,
                    "display_name": display_name,
                    "run_name": run_name,
                    "stable_name": stable_name,
                    "save_path": str(config.get("rollouts_save_path") or "").rstrip("/"),
                    "created_at": _timestamp(getattr(run, "created_at", None)),
                }
            )
        with self._lock:
            self._catalog = catalog
        return catalog

    def _entry_by_id(self, run_id: str) -> dict[str, Any] | None:
        return next((entry for entry in self._get_catalog() if entry["id"] == run_id), None)

    def _resolve_entry(self, run_name: str, directory: str) -> dict[str, Any] | None:
        """Find the W&B lineage containing one physical rollout attempt.

        Exact names and explicit overrides are authoritative.  For an old
        attempt whose W&B run name was overwritten by a resume, candidates are
        restricted to the same rollout directory and stable experiment/seed
        prefix.  If multiple W&B runs reused that location, creation timestamps
        partition attempts between them.
        """
        override = self.overrides.get(run_name) or self.overrides.get(directory)
        if override:
            entry = self._entry_by_id(override)
            if entry is not None:
                return entry
            return {
                "id": override,
                "display_name": run_name,
                "run_name": run_name,
                "stable_name": split_run_name(run_name)[0],
                "save_path": "",
                "created_at": None,
            }

        catalog = self._get_catalog()
        for key in ("run_name", "display_name"):
            exact = next((entry for entry in catalog if entry[key] and entry[key] == run_name), None)
            if exact is not None:
                return exact

        stable_name, attempt_started = split_run_name(run_name)
        candidates = [
            entry
            for entry in catalog
            if entry["save_path"]
            and entry["save_path"].rsplit("/", 1)[-1] == directory
            and entry["stable_name"] == stable_name
        ]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates or attempt_started is None:
            return None
        eligible = [
            entry for entry in candidates if entry["created_at"] is not None and entry["created_at"] <= attempt_started
        ]
        return max(eligible, key=lambda entry: entry["created_at"]) if eligible else None

    def lineage(self, run_name: str, directory: str) -> dict[str, str] | None:
        """Return stable identity for an attempt, or ``None`` if unresolved."""
        try:
            entry = self._resolve_entry(run_name, directory)
        except Exception as error:
            logger.warning("W&B lineage lookup failed for %s: %s", run_name, error)
            return None
        if entry is None:
            return None
        label = entry["run_name"] or entry["display_name"] or run_name
        return {"id": str(entry["id"]), "name": str(label)}

    def evaluations(
        self,
        run_name: str,
        directory: str,
        *,
        run_id: str | None = None,
        metric: str | None = None,
        artifact_step_offset: int = -1,
    ) -> list[dict[str, Any]]:
        """Return validation scores with both optimizer and artifact step numbers."""
        try:
            # A registry-supplied run ID is authoritative and deliberately skips
            # the expensive/fuzzy project-wide catalog lookup.
            entry = None if run_id else self._resolve_entry(run_name, directory)
            resolved_id = str(run_id or (entry["id"] if entry else ""))
            if not resolved_id:
                return []
            cache_key = (resolved_id, metric, artifact_step_offset)
            with self._lock:
                cached = self._evaluations_cache.get(cache_key)
            if cached is not None:
                return [dict(item) for item in cached]
            disk_key = f"{metric or ''}|{artifact_step_offset}"
            stored = self._disk_get(resolved_id, "evaluations", disk_key)
            if stored is not None:
                with self._lock:
                    self._evaluations_cache[cache_key] = stored
                    self._validation_errors.pop(resolved_id, None)
                    self._validation_fetched_at[resolved_id] = self._disk_entry(resolved_id).get("saved_at")
                return [dict(item) for item in stored]
            run = self._get_api().run(f"{self.path}/{resolved_id}")
            selected_metric = metric or self._pick_metric(run)
            values: dict[int, dict[str, Any]] = {}
            if selected_metric:
                rows = list(run.scan_history(keys=["_step", "training_step", selected_metric]))
                if not rows:
                    rows = list(run.scan_history(keys=["_step", selected_metric]))
                for row in rows:
                    if row.get(selected_metric) is None:
                        continue
                    # Separate evaluators can finish after training has advanced.
                    # They log the checkpoint's explicit training_step while W&B's
                    # append-only _step reflects only the arrival order.
                    logged_step = row.get("training_step", row.get("_step"))
                    if logged_step is None:
                        continue
                    optimizer_step = int(logged_step)
                    values[optimizer_step] = {
                        "artifact_step": optimizer_step + artifact_step_offset,
                        "optimizer_step": optimizer_step,
                        "score": float(row[selected_metric]),
                        "metric": selected_metric,
                        "wandb_run_id": resolved_id,
                    }
            result = [values[step] for step in sorted(values)]
            with self._lock:
                self._evaluations_cache[cache_key] = result
                self._validation_errors.pop(resolved_id, None)
                self._validation_fetched_at[resolved_id] = datetime.datetime.now(datetime.UTC).isoformat()
            self._disk_put(resolved_id, getattr(run, "state", None), "evaluations", disk_key, result)
            return [dict(item) for item in result]
        except Exception as error:
            if "resolved_id" in locals() and resolved_id:
                with self._lock:
                    self._validation_errors[resolved_id] = str(error)
                    self._validation_fetched_at[resolved_id] = datetime.datetime.now(datetime.UTC).isoformat()
            logger.warning("W&B validation lookup failed for %s: %s", run_name, error)
            return []

    def invalidate(self, run_id: str | None = None, *, include_training_metrics: bool = True) -> None:
        """Discard cached validation state and optionally the chart histories."""
        # A terminal run's W&B history is immutable, so a targeted (per-run)
        # invalidation — including the startup force-refresh — keeps its disk
        # entry and only clears the in-memory caches, which repopulate from
        # disk. The full invalidation (the Refresh button, run_id=None) drops
        # the disk cache too, as the escape hatch for e.g. a run resumed under
        # the same W&B id after being cached as crashed.
        if run_id is None:
            self._disk_drop(None)
        elif self._disk_entry(run_id).get("run_state") not in TERMINAL_RUN_STATES:
            self._disk_drop(run_id)
        with self._lock:
            if run_id is None:
                self._evaluations_cache.clear()
                if include_training_metrics:
                    self._training_metrics_cache.clear()
                self._progress_cache.clear()
                self._catalog = None
                self._validation_errors.clear()
                self._validation_fetched_at.clear()
            else:
                for key in [key for key in self._evaluations_cache if key[0] == run_id]:
                    self._evaluations_cache.pop(key, None)
                if include_training_metrics:
                    self._training_metrics_cache.pop(run_id, None)
                self._progress_cache.pop(run_id, None)
                self._validation_errors.pop(run_id, None)
                self._validation_fetched_at.pop(run_id, None)

    def status(self, run_id: str) -> dict[str, Any]:
        """Return observable refresh state for one exact W&B run."""
        with self._lock:
            cached = any(key[0] == run_id for key in self._evaluations_cache)
            return {
                "state": "error" if run_id in self._validation_errors else ("fresh" if cached else "pending"),
                "fetched_at": self._validation_fetched_at.get(run_id),
                "error": self._validation_errors.get(run_id),
            }

    def training_metrics(self, run_id: str) -> dict[str, Any]:
        """Return compact, per-step training curves for the viewer.

        W&B scalar history is the correct source for this overview.  Re-reading
        every multi-gigabyte rollout shard merely to draw small charts is
        both slower and considerably more expensive.
        """
        with self._lock:
            cached = self._training_metrics_cache.get(run_id)
            fetch_lock = self._training_metrics_locks.setdefault(run_id, threading.Lock())
        if cached is not None:
            return self._copy_training_metrics(cached)

        with fetch_lock:
            with self._lock:
                cached = self._training_metrics_cache.get(run_id)
            if cached is not None:
                return self._copy_training_metrics(cached)
            stored = self._disk_get(run_id, "training_metrics", "series")
            if stored is not None:
                result = {"series": stored}
                with self._lock:
                    self._training_metrics_cache[run_id] = result
                return self._copy_training_metrics(result)
            return self._fetch_training_metrics(run_id)

    def _fetch_series(self, run_id: str, key: str, preferences: tuple[str, ...]) -> tuple[str, dict[str, Any]]:
        """Resolve one chart series on the fetch pool with a thread-local client."""
        run = self._run_handle(run_id)
        selected = None
        values: dict[int, float] = {}
        # Shared/resumed W&B runs can retain metric history without
        # exposing that metric in run.summary. Probe the known metric
        # names directly instead of using summary as an availability
        # index. W&B also drops every row when any requested key is
        # absent, so fall back to its internal step for older runs that
        # predate the explicit training_step field.
        for metric in preferences:
            rows = list(run.scan_history(keys=["_step", "training_step", metric]))
            if not rows:
                rows = list(run.scan_history(keys=["_step", metric]))
            for row in rows:
                logged_step = row.get("training_step", row.get("_step"))
                raw_value = row.get(metric)
                if (
                    logged_step is None
                    or raw_value is None
                    or not isinstance(raw_value, Real)
                    or not math.isfinite(float(raw_value))
                ):
                    continue
                values[int(logged_step)] = float(raw_value)
            if values:
                selected = metric
                break
        return key, {
            "metric": selected,
            "points": [{"optimizer_step": step, "value": value} for step, value in sorted(values.items())]
            if selected
            else [],
        }

    def _fetch_rejected_series(self, run_id: str) -> dict[str, dict[str, Any]]:
        return self._rejected_group_series(self._run_handle(run_id))

    def _fetch_training_metrics(self, run_id: str) -> dict[str, Any]:
        try:
            run = self._get_api().run(f"{self.path}/{run_id}")
            run_state = getattr(run, "state", None)
            # Every series is an independent scan_history crawl; running them
            # sequentially made the first chart load take minutes per run.
            futures = [
                self._fetch_pool.submit(self._fetch_series, run_id, key, preferences)
                for key, preferences in TRAINING_METRIC_PREFERENCE.items()
            ]
            rejected_future = self._fetch_pool.submit(self._fetch_rejected_series, run_id)
            series: dict[str, dict[str, Any]] = {}
            for future in futures:
                key, value = future.result()
                series[key] = value
            series.update(rejected_future.result())
            result = {"series": series}
            with self._lock:
                self._training_metrics_cache[run_id] = result
            self._disk_put(run_id, run_state, "training_metrics", "series", series)
            return self._copy_training_metrics(result)
        except Exception as error:
            logger.warning("W&B training-metric lookup failed for %s: %s", run_id, error)
            keys = (*TRAINING_METRIC_PREFERENCE, *REJECTED_GROUP_SERIES)
            return {"series": {key: {"metric": None, "points": []} for key in keys}}

    @staticmethod
    def _copy_training_metrics(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "series": {
                key: {**value, "points": [dict(point) for point in value["points"]]}
                for key, value in payload["series"].items()
            }
        }

    @staticmethod
    def _rejected_group_series(run: Any) -> dict[str, dict[str, Any]]:
        """Derive rejection rates from the group counters logged at each step."""
        metrics = list(REJECTED_GROUP_METRICS.values())
        rows = list(run.scan_history(keys=["_step", "training_step", *metrics]))
        if not rows:
            # Older W&B rows may omit one counter, which makes a multi-key scan
            # return nothing. Read each counter independently in that case.
            values_by_metric: dict[str, dict[int, float]] = {}
            for metric in metrics:
                values: dict[int, float] = {}
                metric_rows = list(run.scan_history(keys=["_step", "training_step", metric]))
                if not metric_rows:
                    metric_rows = list(run.scan_history(keys=["_step", metric]))
                for row in metric_rows:
                    step = row.get("training_step", row.get("_step"))
                    value = row.get(metric)
                    if step is not None and isinstance(value, Real):
                        values[int(step)] = float(value)
                values_by_metric[metric] = values
        else:
            values_by_metric = {metric: {} for metric in metrics}
            for row in rows:
                step = row.get("training_step", row.get("_step"))
                if step is None:
                    continue
                for metric in metrics:
                    value = row.get(metric)
                    if isinstance(value, Real):
                        values_by_metric[metric][int(step)] = float(value)

        rejected = values_by_metric[REJECTED_GROUP_METRICS["rejected"]]
        accepted = values_by_metric[REJECTED_GROUP_METRICS["accepted"]]
        all_zero = values_by_metric[REJECTED_GROUP_METRICS["all_zero"]]
        all_one = values_by_metric[REJECTED_GROUP_METRICS["all_one"]]

        def ratio_points(numerator: dict[int, float], denominator: dict[int, float]) -> list[dict[str, float]]:
            return [
                {"optimizer_step": step, "value": numerator[step] / denominator[step]}
                for step in sorted(numerator.keys() & denominator.keys())
                if denominator[step] > 0
            ]

        sampled = {step: rejected[step] + accepted[step] for step in rejected.keys() & accepted.keys()}
        return {
            "rejected_group_rate": {
                "metric": "batch/filtered_prompts / all sampled prompt groups",
                "points": ratio_points(rejected, sampled),
            },
            "rejected_all_zero_rate": {
                "metric": "batch/filtered_prompts_zero / batch/filtered_prompts",
                "points": ratio_points(all_zero, rejected),
            },
            "rejected_all_one_rate": {
                "metric": "batch/filtered_prompts_solved / batch/filtered_prompts",
                "points": ratio_points(all_one, rejected),
            },
        }

    def progress(self, run_id: str) -> dict[str, Any]:
        """Return the latest explicit optimizer step from the W&B summary."""
        with self._lock:
            cached = self._progress_cache.get(run_id)
        if cached is not None:
            return dict(cached)
        stored = self._disk_get(run_id, "progress", "latest")
        if stored is not None:
            with self._lock:
                self._progress_cache[run_id] = stored
            return dict(stored)

        try:
            run = self._get_api().run(f"{self.path}/{run_id}")
            summary = dict(getattr(run, "summary", None) or {})
            raw_step = summary.get("training_step")
            optimizer_step = int(raw_step) if isinstance(raw_step, Real) and raw_step >= 0 else None
            result = {"optimizer_step": optimizer_step, "run_state": getattr(run, "state", None)}
            with self._lock:
                self._progress_cache[run_id] = result
            self._disk_put(run_id, result["run_state"], "progress", "latest", result)
            return dict(result)
        except Exception as error:
            logger.warning("W&B progress lookup failed for %s: %s", run_id, error)
            return {"optimizer_step": None, "run_state": None}

    def evaluated_steps(self, run_name: str, directory: str) -> list[int]:
        """Backward-compatible list of evaluated artifact steps."""
        return [item["artifact_step"] for item in self.evaluations(run_name, directory)]

    @staticmethod
    def _pick_metric(run: Any) -> str | None:
        summary = getattr(run, "summary", None)
        available = {key for key in (summary.keys() if summary else [])}
        for candidate in EVAL_METRIC_PREFERENCE:
            if candidate in available:
                return candidate
        return next((key for key in sorted(available) if str(key).startswith("eval/")), None)
