"""Observe every scored group before the group filter: per-domain metrics and sampled responses.

MILES reward metrics describe the consumed batch, after the zero-variance group
filter has dropped every all-equal group. For binary rewards the surviving
groups average near 0.5 whatever the policy does, so per-domain learning is
invisible there. The runtime bridge (``configuration/arguments.py``) installs
``observed_filter`` as the dynamic sampling filter and keeps the configured
filter as ``args.prefilter_inner_filter_path``. The observer sees each group
that reached the filter (aborted and missing-reward groups are rejected
earlier by MILES), calls the inner filter unchanged, and accumulates scalars.

``rollout_metrics.log_rollout_data`` drains the accumulator once per update, so
each update reports the groups that reached the filter since the previous one.
In fully async runs these are the groups generated during that window, not the
groups the update trained on.

When ``core.sample_dump_fraction`` is positive, a random slice of groups whose
domain is selected by ``core.sample_dump_domains`` is kept, whole group, and
appended on the same drain to ``<save>/prefilter_samples.jsonl`` with the
response text, reward, finish status, length and verifier diagnostics.
"""

import atexit
import collections
import importlib
import json
import random
import threading
import time
import uuid
from pathlib import Path

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

OBSERVED_FILTER = "open_instruct.miles.rollout.prefilter.observed_filter"
DUMP_FILE = "prefilter_samples.jsonl"
# Verifier targets (for code, the tests) are kept in a dump row up to this size.
MAX_TARGET_CHARS = 65536
METRIC_PREFIX = "rollout/prefilter"


def domain_of(samples) -> str:
    """Verifier names identify the domain, as in the inference-record summary."""
    metadata = getattr(samples[0], "metadata", None) or {}
    names = sorted(
        {
            str(spec.get("name"))
            for spec in metadata.get("verifiers") or []
            if isinstance(spec, dict) and spec.get("name") is not None
        }
    )
    domain = "+".join(names) or metadata.get("source_dataset") or "unknown"
    # A slash would nest the metric name.
    return str(domain).replace("/", "_")


def _flatten(group):
    for item in group:
        if isinstance(item, list):
            yield from item
        else:
            yield item


def _status(sample) -> str:
    status = getattr(sample, "status", None)
    return str(getattr(status, "value", status))


def _graded(sample) -> bool:
    metadata = getattr(sample, "metadata", None)
    return not (isinstance(metadata, dict) and "reward_omission" in metadata)


def _reward(args, sample):
    reward = getattr(sample, "reward", None)
    key = getattr(args, "reward_key", None)
    if key and isinstance(reward, dict):
        reward = reward.get(key)
    if isinstance(reward, bool) or not isinstance(reward, int | float):
        return None
    return float(reward)


def classify(rewards: list[float]) -> str:
    """Group outcome from its graded rewards: all_zero, all_one, constant (other value), mixed or unscored."""
    if not rewards:
        return "unscored"
    if max(rewards) != min(rewards):
        return "mixed"
    if rewards[0] == 0:
        return "all_zero"
    if rewards[0] == 1:
        return "all_one"
    return "constant"


def _keep(output) -> bool:
    return bool(getattr(output, "keep", output))


class _Totals:
    def __init__(self):
        self.groups = 0
        self.outcomes = collections.Counter()
        self.kept = 0
        self.samples = 0
        self.truncated = 0
        self.response_tokens = 0
        self.reward_sum = 0.0
        self.reward_count = 0

    def add(self, outcome, kept, samples, rewards):
        self.groups += 1
        self.outcomes[outcome] += 1
        self.kept += kept
        self.samples += len(samples)
        self.truncated += sum(_status(sample) == "truncated" for sample in samples)
        self.response_tokens += sum(int(getattr(sample, "response_length", 0) or 0) for sample in samples)
        self.reward_sum += sum(rewards)
        self.reward_count += len(rewards)

    def metrics(self, prefix):
        metrics = {f"{prefix}/groups": self.groups, f"{prefix}/samples": self.samples}
        for outcome in ("all_zero", "all_one", "constant", "mixed", "unscored"):
            metrics[f"{prefix}/{outcome}_fraction"] = self.outcomes[outcome] / self.groups
        # Any all-equal scored group: what the zero-variance filter drops.
        metrics[f"{prefix}/all_equal_fraction"] = (
            self.outcomes["all_zero"] + self.outcomes["all_one"] + self.outcomes["constant"]
        ) / self.groups
        metrics[f"{prefix}/kept_fraction"] = self.kept / self.groups
        if self.samples:
            metrics[f"{prefix}/truncated_fraction"] = self.truncated / self.samples
            metrics[f"{prefix}/response_tokens_mean"] = self.response_tokens / self.samples
        if self.reward_count:
            metrics[f"{prefix}/reward_mean"] = self.reward_sum / self.reward_count
        return metrics


class Observer:
    """Process-local accumulator; the filter and the rollout log hook run in the rollout manager."""

    def __init__(self, rng=None):
        self._lock = threading.Lock()
        self._rng = rng or random.Random()
        self._attempt = uuid.uuid4().hex[:12]
        self._totals: dict[str, _Totals] = {}
        self._pending: list[dict] = []
        self._dump_path: Path | None = None
        self._failures = 0
        self._dump_failures = 0
        self._atexit = False

    def observe(self, args, group, output):
        samples = list(_flatten(group))
        if not samples:
            return
        domain = domain_of(samples)
        rewards = [r for sample in samples if _graded(sample) and (r := _reward(args, sample)) is not None]
        outcome = classify(rewards)
        kept = _keep(output)
        row = None
        core = getattr(args, "olmo_core", None)
        fraction = getattr(core, "sample_dump_fraction", 0.0) or 0.0
        if fraction > 0 and self._selected_domain(core, samples) and self._rng.random() < fraction:
            row = dump_row(args, samples, domain=domain, outcome=outcome, output=output, attempt=self._attempt)
        with self._lock:
            self._totals.setdefault(domain, _Totals()).add(outcome, kept, samples, rewards)
            if row is not None and len(self._pending) < getattr(core, "sample_dump_max_groups", 0):
                self._pending.append(row)
                if self._dump_path is None and getattr(args, "save", None):
                    self._dump_path = Path(args.save) / DUMP_FILE
                if not self._atexit:
                    atexit.register(self.flush, None)
                    self._atexit = True

    @staticmethod
    def _selected_domain(core, samples):
        domains = getattr(core, "sample_dump_domains", None)
        if not domains:
            return True
        metadata = getattr(samples[0], "metadata", None) or {}
        names = {spec.get("name") for spec in metadata.get("verifiers") or [] if isinstance(spec, dict)}
        names.add(domain_of(samples))
        return bool(names & set(domains))

    def failed(self):
        with self._lock:
            self._failures += 1
            first = self._failures == 1
        if first:
            logger.exception("Pre-filter observation failed; filtering is unaffected")

    def drain(self, rollout_id=None) -> dict[str, float]:
        """Return metrics since the previous drain and append the sampled groups."""
        with self._lock:
            totals, self._totals = self._totals, {}
            failures, self._failures = self._failures, 0
        metrics = {}
        overall = _Totals()
        for domain, total in sorted(totals.items()):
            metrics.update(total.metrics(f"{METRIC_PREFIX}/{domain}"))
            overall.groups += total.groups
            overall.outcomes.update(total.outcomes)
            overall.kept += total.kept
            overall.samples += total.samples
            overall.truncated += total.truncated
            overall.response_tokens += total.response_tokens
            overall.reward_sum += total.reward_sum
            overall.reward_count += total.reward_count
        if overall.groups:
            metrics.update(overall.metrics(f"{METRIC_PREFIX}/all"))
        if failures:
            metrics[f"{METRIC_PREFIX}/observation_failures"] = failures
        written = self.flush(rollout_id)
        if written:
            metrics[f"{METRIC_PREFIX}/dumped_groups"] = written
        return metrics

    def flush(self, rollout_id=None) -> int:
        with self._lock:
            rows, self._pending = self._pending, []
            path = self._dump_path
        if not rows:
            return 0
        if path is None:
            if self._dump_failures == 0:
                logger.warning("sample_dump_fraction is set but the run has no save directory; dropping samples")
            self._dump_failures += 1
            return 0
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Append: a resumed run adds its own attempt's rows after the earlier ones.
            with path.open("a") as stream:
                for row in rows:
                    stream.write(json.dumps({**row, "logged_rollout_id": rollout_id}, default=str) + "\n")
        except Exception:
            self._dump_failures += 1
            if self._dump_failures == 1:
                logger.exception("Could not append sampled responses to %s", path)
            return 0
        return len(rows)


def _targets(metadata):
    targets = {
        spec.get("name"): spec.get("target")
        for spec in metadata.get("verifiers") or []
        if isinstance(spec, dict) and "target" in spec
    }
    if not targets:
        return {}
    text = json.dumps(targets, default=str)
    if len(text) > MAX_TARGET_CHARS:
        return {"targets_omitted_chars": len(text)}
    return {"targets": targets}


def _versions(sample):
    versions = set()
    for call in getattr(sample, "weight_versions", None) or []:
        for item in call if isinstance(call, list) else [call]:
            version = item.get("version") if isinstance(item, dict) else getattr(item, "version", item)
            if version is not None:
                versions.add(str(version))
    return sorted(versions)


def _metadata(sample):
    metadata = getattr(sample, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def dump_row(args, samples, *, domain, outcome, output, attempt):
    first = samples[0]
    metadata = getattr(first, "metadata", None) or {}
    return {
        "recorded_unix": time.time(),
        "attempt": attempt,
        "domain": domain,
        "verifiers": [spec.get("name") for spec in metadata.get("verifiers") or [] if isinstance(spec, dict)],
        "prompt_id": metadata.get("prepared_sample_id") or metadata.get("sample_id"),
        "source_dataset": metadata.get("source_dataset"),
        "source_row": metadata.get("source_row"),
        "group_index": getattr(first, "group_index", None),
        "group_outcome": outcome,
        "filter_kept": _keep(output),
        "filter_reason": getattr(output, "reason", None),
        "prompt": getattr(first, "prompt", None),
        **_targets(metadata),
        "responses": [
            {
                "index": getattr(sample, "index", None),
                "rollout_id": getattr(sample, "rollout_id", None),
                "reward": _reward(args, sample),
                "graded": _graded(sample),
                "finish": _status(sample),
                "response_tokens": getattr(sample, "response_length", None),
                "policy_versions": _versions(sample),
                "reward_components": _metadata(sample).get("reward_components"),
                "verifier_diagnostics": _metadata(sample).get("verifier_diagnostics"),
                "reward_omission": _metadata(sample).get("reward_omission"),
                "response": getattr(sample, "response", None),
            }
            for sample in samples
        ],
    }


OBSERVER = Observer()
_INNER: dict[str, object] = {}


def _inner_filter(path):
    if path is None:
        return None
    if path not in _INNER:
        module, _, name = path.rpartition(".")
        _INNER[path] = getattr(importlib.import_module(module), name)
    return _INNER[path]


def observed_filter(args, samples, **kwargs):
    """Run the configured group filter unchanged, then record the group it judged."""
    inner = _inner_filter(getattr(args, "prefilter_inner_filter_path", None))
    output = inner(args, samples, **kwargs) if inner is not None else True
    try:
        OBSERVER.observe(args, samples, output)
    except Exception:
        OBSERVER.failed()
    return output


def drain(rollout_id=None) -> dict[str, float]:
    return OBSERVER.drain(rollout_id)
