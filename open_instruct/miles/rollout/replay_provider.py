"""Checkpointed low-success history and length-triggered fresh replay.

The base provider still chooses ordinary prompts. This optional layer reserves a
fixed fraction of prompt draws for low-success prompts after a within-source drop in
recent response length. It never reuses generations or changes verifier rewards.
"""

import copy
import hashlib
import json
from collections import Counter, deque
from fractions import Fraction

from open_instruct.miles.configuration import validation
from open_instruct.miles.datasets import replay_evidence


def identity(sample):
    metadata = sample.metadata or {}
    source = metadata.get("mixture_source") or metadata.get("source_dataset") or "dataset"
    verifiers = [{"name": v.get("name"), "target": v.get("target")} for v in metadata.get("verifiers", [])]
    # Match native inference-record task identity, including every rendered turn.
    raw = json.dumps(
        {"basis": "rendered", "prompt": sample.prompt, "verifiers": verifiers},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    task = hashlib.sha256(raw.encode()).hexdigest()
    return f"{source}:{task}", source, task


class ReplayPromptProvider:
    requires_feedback = True

    def __init__(
        self,
        base,
        samples,
        *,
        fraction=0.1,
        length_drop=0.1,
        window=256,
        max_successes=1,
        truncation="any",
        dataset_guard=None,
    ):
        validation.number(fraction, "prompt_replay_fraction", maximum=1)
        validation.number(length_drop, "prompt_replay_length_drop", maximum=1)
        if length_drop == 1:
            raise ValueError("prompt_replay_length_drop must be below 1")
        validation.integer(window, "prompt_replay_window", minimum=1)
        validation.integer(max_successes, "prompt_replay_max_successes", minimum=0)
        validation.choice(truncation, "prompt_replay_truncation", ("none", "any", "half", "all"))
        self.base = base
        self.dataset_guard = dataset_guard
        self.config = dict(
            fraction=fraction,
            length_drop=length_drop,
            window=window,
            max_successes=max_successes,
            truncation=truncation,
        )
        self.fraction = Fraction(str(fraction))
        self.samples = {}
        self.sources = {}
        self.tasks = {}
        for sample in samples:
            key, source, task = identity(sample)
            self.samples.setdefault(key, sample)
            self.sources[key], self.tasks[key] = source, task
        if not self.samples:
            raise ValueError("Prompt replay requires a nonempty prepared prompt dataset")
        self.catalog = hashlib.sha256(json.dumps(sorted(self.samples)).encode()).hexdigest()
        self.draws = 0
        self.pending = {}
        self.inflight = set()
        self.history = {}
        self.windows = {}
        self.source_counts = Counter()
        self.counts = Counter()

    def _check_dataset(self):
        if self.dataset_guard is not None and not self.dataset_guard():
            raise RuntimeError("Prompt replay dataset cannot be replaced after construction")

    def _mean(self, source):
        rows = self.windows.get(source, [])
        n = sum(r[1] for r in rows)
        return sum(r[0] for r in rows) / n if n else None

    def _eligible(self):
        result = []
        means = {source: self._mean(source) for source in self.windows}
        for key, entry in self.history.items():
            if not entry["candidate"] or key in self.inflight:
                continue
            source = self.sources[key]
            if self.source_counts[source] - entry["at_count"] < self.config["window"]:
                continue
            current = means.get(source)
            if current is not None and current <= entry["baseline_length"] * (1 - self.config["length_drop"]):
                result.append(key)
        return result

    def available(self):
        if getattr(self.base, "available", lambda: True)():
            return True
        numerator, denominator = self.fraction.numerator, self.fraction.denominator
        slot = (self.draws + 1) * numerator // denominator > self.draws * numerator // denominator
        return bool(slot and self._eligible())

    def next_samples(self, n):
        self._check_dataset()
        if type(n) is not int or n < 0:
            raise ValueError("n must be a nonnegative integer")
        eligible = deque(self._eligible()) if n and self.fraction else deque()
        result = []
        numerator, denominator = self.fraction.numerator, self.fraction.denominator
        for _ in range(n):
            slot = (self.draws + 1) * numerator // denominator > self.draws * numerator // denominator
            review = bool(slot and eligible)
            if review:
                key = eligible.popleft()
                sample = self.samples[key]
                self.inflight.add(key)
            else:
                sample = self.base.next_samples(1)[0]
                key = identity(sample)[0]
                if key not in self.samples:
                    raise RuntimeError("Prompt replay dataset changed after construction")
            sample = copy.copy(sample)
            sample.metadata = (sample.metadata or {}) | {
                replay_evidence.SELECTION_ID: self.draws,
                replay_evidence.REPLAY: review,
                replay_evidence.KEY: key,
            }
            self.pending[self.draws] = (key, review)
            self.draws += 1
            result.append(sample)
        return result

    def feedback(self, records):
        self._check_dataset()
        for record in records:
            selection_id = record.get(replay_evidence.SELECTION_ID)
            if record.get("kind") == "disposition" and record.get("disposition") == "acknowledged":
                pending = self.pending.pop(selection_id, None)
                if pending and pending[1]:
                    self.inflight.discard(pending[0])
                continue
            if record.get("kind") != "group" or record.get("filter_decision") == "aborted":
                continue
            pending = self.pending.pop(selection_id, None)
            if pending is None:
                continue  # First scored generation counts; stale regeneration is a duplicate.
            key, review = pending
            if review:
                self.inflight.discard(key)
            outcome = replay_evidence.classify(record)
            category = outcome["category"]
            source = self.sources[key]
            # Fresh draws alone estimate length: selecting truncated replay prompts
            # must not inflate the baseline that decides when to replay again.
            if not review and category not in ("invalid", "aborted") and outcome["response_tokens"] is not None:
                self.windows.setdefault(source, deque(maxlen=self.config["window"])).append(
                    (outcome["response_tokens"], outcome["group_size"])
                )
                self.source_counts[source] += 1
            entry = self.history.setdefault(
                key,
                {
                    "source": source,
                    "task_key": self.tasks[key],
                    "counts": {},
                    "candidate": False,
                    "baseline_length": None,
                    "at_count": 0,
                },
            )
            entry["counts"][category] = entry["counts"].get(category, 0) + 1
            self.counts[category] += 1
            self.counts[f"source/{source}/{category}"] += 1
            entry["last_outcome"] = outcome
            if category not in ("invalid", "aborted") and outcome["binary_successes"] is not None:
                bucket = f"successes_{int(outcome['binary_successes'])}_of_{outcome['group_size']}"
                caps = f"truncated_{outcome['truncated']}_of_{outcome['group_size']}"
                self.counts[bucket] += 1
                entry["counts"][f"{bucket}/{caps}"] = entry["counts"].get(f"{bucket}/{caps}", 0) + 1
                if review:
                    self.counts[f"replay_{bucket}"] += 1
            for response in record.get("responses", []):
                probe = response.get("literal_reference_in_text")
                if type(probe) is bool:
                    self.counts["literal_reference_checked"] += 1
                    self.counts["literal_reference_found"] += probe
                    if category in replay_evidence.ZERO_CATEGORIES:
                        self.counts["zero_literal_reference_checked"] += 1
                        self.counts["zero_literal_reference_found"] += probe
                    if response.get("truncated"):
                        self.counts["truncated_literal_reference_checked"] += 1
                        self.counts["truncated_literal_reference_found"] += probe
            if review:
                self.counts["replayed"] += 1
                self.counts[f"replay_{category}"] += 1
                self.counts[f"replay_disposition/{record.get('filter_decision', 'unknown')}"] += 1
                self.counts[f"source/{source}/replay_{category}"] += 1
                entry["counts"]["replayed"] = entry["counts"].get("replayed", 0) + 1
            matches = replay_evidence.eligible(
                outcome, max_successes=self.config["max_successes"], truncation=self.config["truncation"]
            )
            if category != "invalid":
                entry["candidate"] = matches
            # Every retry waits for a fresh full window and another relative drop.
            # An invalid replay remains eligible later, without a tight retry loop.
            if matches or review:
                mean = self._mean(source)
                entry["baseline_length"] = mean
                entry["at_count"] = self.source_counts[source]
                entry["candidate"] = entry["candidate"] and mean is not None
            entry["last_observation_id"] = record.get("observation_id")
            entry["last_policy_versions"] = sorted(
                {str(v) for r in record.get("responses", []) for v in r.get("policy_versions", [])}
            )
            record["prompt_replay"] = outcome | {
                "replay": review,
                "prompt_key": key,
                "candidate": entry["candidate"],
                "baseline_length": entry["baseline_length"],
                "source_mean_length": self._mean(source),
            }

    def metrics(self):
        return {f"rollout/prompt_replay/{k}": v for k, v in self.counts.items()} | {
            "rollout/prompt_replay/prompts": len(self.history),
            "rollout/prompt_replay/candidates": sum(r["candidate"] for r in self.history.values()),
            "rollout/prompt_replay/inflight": len(self.inflight),
        }

    def selection_state_dict(self):
        self._check_dataset()
        # Delivery rollback cannot include feedback (caller holds its lock). Avoid
        # copying the whole prompt history on every N-next request.
        return {
            "base": self.base.state_dict(),
            "draws": self.draws,
            "pending": dict(self.pending),
            "inflight": sorted(self.inflight),
        }

    def load_selection_state_dict(self, state):
        self.base.load_state_dict(state["base"])
        self.draws, self.pending, self.inflight = state["draws"], dict(state["pending"]), set(state["inflight"])

    def state_dict(self):
        return self.selection_state_dict() | {
            "kind": "prompt_replay",
            "config": dict(self.config),
            "catalog": self.catalog,
            "history": copy.deepcopy(self.history),
            "windows": {k: list(v) for k, v in self.windows.items()},
            "source_counts": dict(self.source_counts),
            "counts": dict(self.counts),
        }

    def load_state_dict(self, state):
        self._check_dataset()
        for field, expected in (("kind", "prompt_replay"), ("config", self.config), ("catalog", self.catalog)):
            if state.get(field) != expected:
                raise RuntimeError(
                    f"Prompt replay checkpoint mismatch for {field}: checkpoint={state.get(field)!r}, current={expected!r}"
                )
        keys = set(state["history"]) | {v[0] for v in state["pending"].values()} | set(state["inflight"])
        if not keys <= self.samples.keys():
            raise RuntimeError("Prompt replay checkpoint contains unknown prompt IDs")
        expected_inflight = {key for key, review in state["pending"].values() if review}
        if set(state["inflight"]) != expected_inflight:
            raise RuntimeError("Prompt replay checkpoint has inconsistent in-flight reviews")
        if (
            type(state["draws"]) is not int
            or state["draws"] < 0
            or any(type(i) is not int or not 0 <= i < state["draws"] for i in state["pending"])
        ):
            raise RuntimeError("Prompt replay checkpoint has invalid selection counters")
        for entry in state["history"].values():
            if entry["candidate"] and (
                not isinstance(entry["baseline_length"], (int, float)) or entry["baseline_length"] < 0
            ):
                raise RuntimeError("Prompt replay checkpoint has a candidate without a valid length baseline")
        self.load_selection_state_dict(state)
        self.history = copy.deepcopy(state["history"])
        self.windows = {k: deque(v, maxlen=self.config["window"]) for k, v in state["windows"].items()}
        self.source_counts, self.counts = Counter(state["source_counts"]), Counter(state["counts"])
