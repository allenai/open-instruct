#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapt the MILES buffered data source for prompt selection and recoverable runs.
The wrapper skips prompts named in a frozen exclusion table and tracks outstanding
prompt groups alongside the dataset cursor, so retries and resumes regenerate
unconsumed work rather than losing it or reusing partial responses. It also waits
for dashboard telemetry to drain before the rollout actor shuts down.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import ray
import torch
from miles.backends.core_utils.rollout import response_seed
from miles.dashboard import backend, hooks
from miles.utils.types import Sample

from miles.rollout.data_source import RolloutDataSourceWithBuffer
from open_instruct import logger_utils
from open_instruct.miles.datasets import ads, record_selection

logger = logger_utils.setup_logger(__name__)


class DashboardDrainingRolloutDataSource:
    """Delegate to MILES' buffered data source and drain telemetry on close.

    MILES calls ``data_source.close()`` in the rollout-manager actor before the
    driver shuts down the dashboard collector. That is the only lifecycle seam
    needed to turn the trajectory sink's fire-and-forget flush into a durable
    actor barrier, without patching MILES itself.

    Args:
        args: Parsed MILES runtime arguments.

    """

    def __init__(self, args: Any) -> None:
        """Initialize the MILES buffered-data-source delegate."""

        self._delegate = RolloutDataSourceWithBuffer(args)
        self._cursor_lock = threading.RLock()
        self._pending_groups: dict[int, Any] = {}
        self._recovery_groups: list[Any] | None = None
        self._fully_async = bool(getattr(args, "fully_async", False))
        self._collect_dashboard = bool(getattr(args, "use_miles_dashboard", False))
        core = getattr(args, "olmo_core", None)
        self._selection = record_selection.Selection(args) if getattr(core, "selection_table", None) else None
        self._ads = None
        if getattr(core, "ads_table", None):
            raw = Path(core.ads_table).read_bytes()
            if hashlib.sha256(raw).hexdigest() != core.ads_sha256:
                raise ValueError("ADS table digest mismatch")
            table = [json.loads(line) for line in raw.splitlines()]
            self._ads = ads.Schedule(
                table, seed=args.rollout_seed, mini_size=core.ads_mini_size, active_clusters=core.ads_active_clusters
            )
            self._ads_samples = {self._ads_identity(sample): sample for sample in self._delegate.dataset.samples}
            if len(self._ads_samples) != len(self._delegate.dataset.samples) or set(self._ads_samples) != set(
                self._ads.lookup
            ):
                raise ValueError("ADS table must cover exactly the unfiltered training pool")
        self._opener_seeds = tuple(getattr(core, "opener_seeds", ()) or ())
        self._opener_seeds_per_group = getattr(core, "opener_seeds_per_group", 0)

    def __getattr__(self, name: str) -> Any:
        """Delegate attributes not implemented by this lifecycle adapter."""
        return getattr(self._delegate, name)

    def get_samples(self, num_samples: int) -> Any:
        """Return samples from the underlying MILES data source."""
        with self._cursor_lock:
            if self._ads is not None:
                groups = self._take_ads(num_samples)
            elif self._selection is None:
                groups = self._delegate.get_samples(num_samples)
            else:
                # Skipped prompts never enter the restart ledger; the saved cursor moves past
                # them, and the frozen table skips the same prompts again after a resume.
                dataset = getattr(self._delegate, "dataset", None)
                limit = len(dataset) if dataset is not None else 1_000_000
                groups = record_selection.take(self._delegate.get_samples, num_samples, self._selection.keep, limit)
            self._seed_openers(groups)
            if self._recovery_groups is not None:
                self._recovery_groups.extend(copy.deepcopy(groups))
            for group in groups:
                identity = group[0].group_index
                if identity not in self._pending_groups:
                    self._pending_groups[identity] = copy.deepcopy(group)
            return groups

    @staticmethod
    def _ads_identity(sample):
        meta = sample.metadata
        return f"{meta['prepared_sample_id']}:{meta['prompt_format']}"

    def _take_ads(self, count):
        delegate = self._delegate
        groups = delegate._get_samples_from_buffer(count)
        for _ in range(count - len(groups)):
            identity, round_id, epoch = self._ads.take()
            group = []
            for _ in range(delegate.args.n_samples_per_prompt):
                sample = copy.deepcopy(self._ads_samples[identity])
                sample.group_index, sample.index = delegate.sample_group_index, delegate.sample_index
                sample.metadata = {
                    **sample.metadata,
                    "ads": {"round": round_id, "epoch": epoch, "cluster": self._ads.lookup[identity]},
                }
                delegate.sample_index += 1
                group.append(sample)
            delegate.sample_group_index += 1
            delegate.sample_offset += 1
            groups.append(group)
        delegate.epoch_id = self._ads.epoch
        return groups

    def observe_ads(self, group):
        """Consume complete binary reward groups once, including zero-variance groups."""
        with self._cursor_lock:
            identity = group[0].group_index
            pending = self._pending_groups[identity]
            if pending[0].metadata.get("ads_feedback_seen"):
                return
            rewards = [sample.get_reward_value(self._delegate.args) for sample in group]
            if any(not isinstance(r, (int, float)) or not math.isfinite(r) or r not in (0, 1) for r in rewards):
                raise ValueError("ADS pilot requires complete binary rewards")
            self._ads.observe(
                self._ads_identity(group[0]), group[0].metadata["ads"]["round"], sum(rewards) / len(rewards)
            )
            for sample in pending:
                sample.metadata["ads_feedback_seen"] = True

    def _seed_openers(self, groups: Any) -> None:
        """Request opening text for the first samples of each new group.

        Openers rotate across groups by prompt identity, so every opener is tried
        on a fixed share of each collection and retries reproduce the assignment.
        The request lives in sample metadata and survives the pristine-prompt
        ledger, resets and resume; the refresh producer scores and applies it.
        """
        if not self._opener_seeds:
            return
        count, openers = self._opener_seeds_per_group, self._opener_seeds
        for group in groups:
            for slot, sample in enumerate(group[:count]):
                text = openers[(group[0].group_index * count + slot) % len(openers)]
                sample.metadata = {**(sample.metadata or {}), response_seed.REQUEST_KEY: {"text": text}}

    def persist_group_metadata(self, group: Any, keys: tuple[str, ...]) -> None:
        """Copy producer-chosen metadata into this group's pristine retry and resume records.

        The refresh producer decides opener branches after the group is handed out;
        writing them back means a requeued or resumed group reuses the same draw.
        """
        identity = group[0].group_index
        with self._cursor_lock:
            records = [self._pending_groups[identity]] if identity in self._pending_groups else []
            records += [g for g in self._recovery_groups or () if g[0].group_index == identity]
            if not records:
                raise RuntimeError(f"prompt group {identity} has no pristine record to update")
            for record in records:
                if [s.index for s in record] != [s.index for s in group]:
                    raise RuntimeError(f"pristine record for prompt group {identity} does not match its samples")
                for pristine, sample in zip(record, group, strict=True):
                    chosen = {key: copy.deepcopy(sample.metadata[key]) for key in keys}
                    pristine.metadata = {**(pristine.metadata or {}), **chosen}

    def add_samples(self, samples: Any) -> None:
        """Return aborted samples to the underlying MILES buffer."""
        with self._cursor_lock:
            self._delegate.add_samples(samples)
            if not self._fully_async:
                # Sync generation returns its unused groups after joining all work.
                # Everything else was consumed or intentionally discarded.
                self._pending_groups = self._sync_buffered_groups()

    def _sync_buffered_groups(self) -> dict[int, Any]:
        """Keep pristine prompts in buffer order, never the partial responses."""
        groups = {}
        for group in self._delegate.buffer:
            identity = group[0].group_index
            if identity not in self._pending_groups:
                raise RuntimeError(f"sync buffered prompt group {identity} has no pristine prompt record")
            if identity in groups:
                raise RuntimeError(f"duplicate sync buffered prompt group {identity}")
            groups[identity] = self._pending_groups[identity]
        return groups

    def begin_recovery_batch(self) -> None:
        """Retain pristine prompts until this synchronous batch completes."""
        with self._cursor_lock:
            if self._fully_async or self._recovery_groups is not None:
                raise RuntimeError("recovery batch requires an idle synchronous data source")
            self._recovery_groups = []

    def recovery_group_indices(self) -> list[int]:
        """Expose acquired prompt identities for recovery evidence."""
        with self._cursor_lock:
            if self._recovery_groups is None:
                raise RuntimeError("no recovery batch is active")
            return [group[0].group_index for group in self._recovery_groups]

    def finish_recovery_batch(self, *, retry: bool) -> int:
        """Requeue all acquired groups after the generation thread has joined.

        The dataset cursor stays advanced: these exact identities now live in
        the buffer. Generated responses and partially completed groups are never
        reused, and no acquired prompt is silently skipped by a recovery retry.
        """
        with self._cursor_lock:
            if self._recovery_groups is None:
                raise RuntimeError("no recovery batch is active")
            groups = self._recovery_groups
            if retry:
                identities = [group[0].group_index for group in groups]
                if len(set(identities)) != len(identities):
                    raise RuntimeError("duplicate prompt identity in recovery batch")
                self._delegate.buffer = groups + self._delegate.buffer
            self._recovery_groups = None
            return len(groups)

    def acknowledge_groups(self, groups: Any) -> None:
        """Retire consumed or intentionally dropped prompts from the restart ledger."""
        if self._fully_async:
            with self._cursor_lock:
                for group in groups:
                    identity = group[0].group_index
                    if identity not in self._pending_groups:
                        raise RuntimeError(f"async prompt group {identity} was not outstanding")
                    del self._pending_groups[identity]

    def requeue_pending_groups(self, identities: list[int]) -> int:
        """Reset joined, unconsumed async work to its pristine checkpoint ledger."""
        with self._cursor_lock:
            identities = list(dict.fromkeys(identities))
            if not self._fully_async or any(i not in self._pending_groups for i in identities):
                raise RuntimeError("cannot requeue an untracked async prompt group")
            selected = set(identities)
            groups = [copy.deepcopy(self._pending_groups[i]) for i in identities]
            self._delegate.buffer = groups + [g for g in self._delegate.buffer if g[0].group_index not in selected]
            return len(groups)

    def save(self, rollout_id: int) -> None:
        """Atomically save the cursor and pristine outstanding prompts together.

        Generation runs on another event-loop thread. Keep one lock across the
        cursor snapshot and prompt ledger; no generated response is checkpointed.
        Resume regenerates pending groups under the restored actor policy.
        Sync saves run between collections and retain only buffered groups;
        async saves also retain in-flight and completed-but-unconsumed groups.
        """

        delegate = self._delegate
        if not delegate.args.rollout_global_dataset:
            return
        with self._cursor_lock:
            state = {
                key: getattr(delegate, key)
                for key in ("sample_offset", "epoch_id", "sample_group_index", "sample_index", "metadata")
            }
            if self._ads is not None:
                state["ads"] = self._ads.state_dict()
            if not self._fully_async:
                self._pending_groups = self._sync_buffered_groups()
            key = "olmo_async_pending" if self._fully_async else "olmo_sync_pending"
            state[key] = {
                "schema_version": 1,
                "groups": [[sample.to_dict() for sample in group] for group in self._pending_groups.values()],
            }
            destination = Path(delegate.args.save) / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=".cursor-", dir=destination.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    torch.save(state, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, destination)
                directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                Path(name).unlink(missing_ok=True)

    def load(self, rollout_id: int | None = None) -> None:
        """Restore the cursor and regenerate pending work from pristine prompts."""
        with self._cursor_lock:
            self._delegate.load(rollout_id)
            if not self._delegate.args.rollout_global_dataset or not self._delegate.args.load:
                return
            mode = "async" if self._fully_async else "sync"
            path = Path(self._delegate.args.load) / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
            # Initial actor checkpoints have no RL cursor. A resumed RL attempt must.
            if not path.exists():
                if self._fully_async and getattr(self._delegate.args, "start_rollout_id", 0) > 0:
                    raise RuntimeError("async resume requires a complete prompt cursor")
                return

            state = torch.load(path, map_location="cpu", weights_only=True)
            if self._ads is not None:
                if "ads" not in state:
                    raise ValueError("ADS resume checkpoint lacks sampler state")
                self._ads.load_state_dict(state["ads"])
            key = f"olmo_{mode}_pending"
            if not self._fully_async and key not in state:
                logger.warning(
                    "Legacy sync cursor has no pending-prompt ledger; restoring the cursor only. "
                    "Any buffered prompts from that checkpoint cannot be recovered."
                )
                self._pending_groups = {}
                self._delegate.buffer = []
                return
            pending = state.get(key)
            if not isinstance(pending, dict) or pending.get("schema_version") != 1:
                raise RuntimeError(f"{mode} resume cursor lacks the pending-prompt ledger; unsafe legacy checkpoint")
            groups = [[Sample.from_dict(sample) for sample in group] for group in pending["groups"]]
            restored = {}
            for group in groups:
                if (
                    len(group) != self._delegate.args.n_samples_per_prompt
                    or len({sample.group_index for sample in group}) != 1
                    or group[0].group_index in restored
                    or any(sample.weight_versions or sample.response_length for sample in group)
                ):
                    raise RuntimeError(f"{mode} resume cursor has an invalid pending prompt group")
                restored[group[0].group_index] = copy.deepcopy(group)
            self._pending_groups = restored
            self._delegate.buffer = groups
            logger.info("Restored %d pending %s prompt groups for regeneration", len(groups), mode)

    def get_buffer_length(self) -> int | None:
        """Return the number of buffered sample groups."""
        return self._delegate.get_buffer_length()

    def close(self) -> None:
        """Flush rollout telemetry and wait until the collector persists it."""
        if (close := getattr(self._delegate, "close", None)) is not None:
            close()
        if not self._collect_dashboard:
            return

        handle = backend.current_collector()
        hooks.detach_and_flush()
        if handle is None:
            logger.warning("dashboard collector unavailable while draining rollout telemetry")
            return
        ray.get(handle.flush.remote(), timeout=30)
        logger.info("Drained rollout dashboard telemetry before collector shutdown")
