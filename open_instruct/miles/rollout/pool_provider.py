"""Finite named pools with independent selection, accounting and optional replay."""

import copy
import math
import random
from collections import Counter

from open_instruct.miles.datasets import replay_evidence
from open_instruct.miles.rollout import indexed_samples, pool_metrics, pool_sampling, replay_provider

SELECTION_ID = "pool_selection_id"
POOL = "sampling_pool"
RECEIPT_FIELDS = (
    "origin_id",
    "source_dataset",
    "source_row",
    "mixture_source",
    "release_id",
    "record_id",
    "labeling_version",
    "prepared_sample_id",
    "sample_id",
    "pool_id",
    "sampling_pool",
)


class FinitePrompts:
    requires_feedback = False

    def __init__(self, samples, *, seed, shuffle):
        self.samples = samples.ordered(seed, shuffle) if isinstance(samples, indexed_samples.View) else list(samples)
        if shuffle and not isinstance(samples, indexed_samples.View):
            random.Random(seed).shuffle(self.samples)
        self.offset = 0
        if isinstance(self.samples, indexed_samples.View):
            self.samples.prefetch(0)

    def available(self):
        return self.offset < len(self.samples)

    def next_samples(self, n):
        if n > len(self.samples) - self.offset:
            raise StopIteration("Fresh pool exhausted")
        result = self.samples[self.offset : self.offset + n]
        self.offset += n
        if isinstance(self.samples, indexed_samples.View):
            self.samples.prefetch(self.offset)
        return result

    def state_dict(self):
        return {"offset": self.offset}

    def load_state_dict(self, state):
        offset = state["offset"]
        if type(offset) is not int or not 0 <= offset <= len(self.samples):
            raise ValueError("Invalid finite pool cursor")
        self.offset = offset
        if isinstance(self.samples, indexed_samples.View):
            self.samples.prefetch(self.offset)


class PoolPromptProvider:
    requires_feedback = True

    def __init__(self, samples, *, pools, seed, shuffle, identity, replay=None, sampling=None):
        self.sampling = pool_sampling.validate({} if sampling is None else sampling, pools)
        self.identity = copy.deepcopy(
            {
                "version": 4,
                "manifest": identity,
                "pools": pools,
                "seed": seed,
                "shuffle": shuffle,
                "replay": replay,
                "sampling": self.sampling,
            }
        )
        self.rng = random.Random(seed)
        # A single permutation fixes fresh global order independently of outcomes.
        # Replays can be inserted before a fresh draw without advancing this order.
        self.catalog = samples if isinstance(samples, indexed_samples.Catalog) else None
        if self.catalog is not None:
            self.fresh_order, pool_members = self.catalog.split(self.sampling["policy"], shuffle, seed)
        else:
            samples = list(samples)
            if self.sampling["policy"] == "shuffle" and shuffle:
                random.Random(seed).shuffle(samples)
            self.fresh_order = [s.metadata[POOL] for s in samples] if self.sampling["policy"] == "shuffle" else []
            if any(s.metadata[POOL] not in pools for s in samples):
                raise ValueError("Unknown runtime pool")
            pool_members = {name: [s for s in samples if s.metadata[POOL] == name] for name in pools}
        self.providers = {}
        for name in pools:
            members = pool_members[name]
            base = FinitePrompts(
                members, seed=f"{seed}:{name}", shuffle=shuffle and self.sampling["policy"] != "shuffle"
            )
            self.providers[name] = (
                replay_provider.ReplayPromptProvider(base, members, **replay) if replay and members else base
            )
        self.draws = 0
        self.pending = {}
        self.trained = Counter()
        self.targets = Counter()
        self.terminal = Counter()
        self.proposed = Counter()
        self.replay_proposed = Counter()
        self.observations = Counter()
        self.tokens = Counter()
        self.committed_update = -1

    def close(self):
        if self.catalog is not None:
            self.catalog.close()

    def active(self):
        return [name for name, provider in self.providers.items() if provider.available()]

    def available(self):
        return bool(self.active())

    def _choose_pool(self, active):
        if self.sampling["policy"] == "shuffle":
            offset = sum(getattr(p, "base", p).offset for p in self.providers.values())
            if offset < len(self.fresh_order):
                return self.fresh_order[offset]
        weights = pool_sampling.draw_weights(
            self.sampling,
            active,
            trained=self.trained,
            targets=self.targets,
            terminal=self.terminal,
            pending=self.pending,
        )
        return self.rng.choices(active, weights=weights, k=1)[0]

    def next_samples(self, n):
        if type(n) is not int or n < 0:
            raise ValueError("n must be a nonnegative integer")
        result = []
        for _ in range(n):
            active = self.active()
            if not active:
                raise StopIteration("All named pools exhausted")
            name = self._choose_pool(active)
            sample = copy.copy(self.providers[name].next_samples(1)[0])
            sample.metadata = sample.metadata | {SELECTION_ID: self.draws}
            # Pending accounting must not retain or deepcopy verifier payloads.
            metadata = {
                key: sample.metadata[key]
                for key in (*RECEIPT_FIELDS, replay_evidence.SELECTION_ID, replay_evidence.REPLAY, replay_evidence.KEY)
                if key in sample.metadata
            }
            self.pending[self.draws] = {"pool": name, "active": active, "metadata": metadata}
            self.draws += 1
            self.proposed[name] += 1
            self.replay_proposed[name] += bool(sample.metadata.get("prompt_replay_review", False))
            result.append(sample)
        return result

    def feedback(self, records):
        for record in records:
            entry = self.pending.get(record.get(SELECTION_ID))
            if entry is None:
                continue
            if record.get("kind") == "group" and not entry.get("observed"):
                entry["observed"] = True
                pool_metrics.observe(self.observations, entry, record)
                lengths = [r.get("response_tokens") for r in record.get("responses", [])]
                if lengths and all(type(n) is int and n >= 0 for n in lengths):
                    entry["response_tokens"] = sum(lengths)
                    self.tokens[f"{entry['pool']}/generated_tokens"] += sum(lengths)
                else:
                    self.tokens[f"{entry['pool']}/generated_tokens_unknown_groups"] += 1
            provider = self.providers[entry["pool"]]
            if isinstance(provider, replay_provider.ReplayPromptProvider):
                provider.feedback(
                    [
                        record
                        | entry["metadata"]
                        | ({"disposition": "acknowledged"} if record.get("disposition") == "discarded" else {})
                    ]
                )
            if (record.get("kind") == "disposition" and record.get("disposition") == "discarded") or (
                record.get("kind") == "group" and record.get("filter_decision") == "filtered"
            ):
                if record.get("kind") == "disposition":
                    self.observations[f"{entry['pool']}/discarded"] += 1
                    self.observations[f"{entry['pool']}/discarded_unobserved"] += not entry.get("observed", False)
                self.pending.pop(record[SELECTION_ID])
                self.terminal[entry["pool"]] += 1

    def commit(self, update, selections):
        """Credit a completed optimizer update exactly once, not queue acceptance."""
        if update <= self.committed_update:
            if update == self.committed_update and selections == getattr(self, "last_commit", None):
                return
            raise ValueError("Out-of-order or conflicting exposure commit")
        if len(set(selections)) != len(selections) or not set(selections) <= self.pending.keys():
            raise ValueError("Unknown or duplicate selections in trained exposure")
        for selection in selections:
            entry = self.pending.pop(selection)
            name = entry["pool"]
            self.trained[name] += 1
            if "response_tokens" in entry:
                self.tokens[f"{name}/trained_tokens"] += entry["response_tokens"]
            else:
                self.tokens[f"{name}/trained_tokens_unknown_groups"] += 1
            self.terminal[name] += 1
            replay = entry["metadata"].get("prompt_replay_review", False)
            self.tokens[f"{name}/{'replay' if replay else 'fresh'}_groups"] += 1
            if self.sampling["policy"] == "exposure":
                for active in entry["active"]:
                    self.targets[active] += 1 / len(entry["active"])
        self.committed_update = update
        self.last_commit = list(selections)

    def state_dict(self):
        return {
            "kind": "pools",
            "identity": copy.deepcopy(self.identity),
            "rng": self.rng.getstate(),
            "providers": {p: v.state_dict() for p, v in self.providers.items()},
            "draws": self.draws,
            "pending": copy.deepcopy(self.pending),
            "trained": dict(self.trained),
            "targets": dict(self.targets),
            "terminal": dict(self.terminal),
            "proposed": dict(self.proposed),
            "replay_proposed": dict(self.replay_proposed),
            "observations": dict(self.observations),
            "tokens": dict(self.tokens),
            "committed_update": self.committed_update,
            "last_commit": getattr(self, "last_commit", None),
        }

    def load_state_dict(self, state):
        if state.get("kind") != "pools" or state.get("identity") != self.identity:
            raise ValueError("Pool checkpoint identity mismatch")
        if set(state["providers"]) != set(self.providers):
            raise ValueError("Pool checkpoint membership mismatch")
        draws = state["draws"]
        if (
            type(draws) is not int
            or draws < 0
            or any(
                type(i) is not int
                or not 0 <= i < draws
                or entry["pool"] not in self.providers
                or not entry["active"]
                or not set(entry["active"]) <= self.providers.keys()
                for i, entry in state["pending"].items()
            )
        ):
            raise ValueError("Invalid pool checkpoint pending selections")
        for field in ("trained", "targets", "terminal", "proposed", "replay_proposed"):
            if not set(state[field]) <= self.providers.keys() or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in state[field].values()
            ):
                raise ValueError("Invalid pool checkpoint counters")
        observation_keys = {f"{pool}/{key}" for pool in self.providers for key in pool_metrics.OBSERVATIONS}
        if not set(state["observations"]) <= observation_keys or any(
            type(value) is not int or value < 0 for value in state["observations"].values()
        ):
            raise ValueError("Invalid pool checkpoint observations")
        if sum(state["terminal"].values()) + len(state["pending"]) != draws:
            raise ValueError("Pool checkpoint reservations do not reconcile")
        random.Random().setstate(state["rng"])
        previous = {p: v.state_dict() for p, v in self.providers.items()}
        try:
            for name, provider in self.providers.items():
                provider.load_state_dict(state["providers"][name])
        except Exception:
            for name, provider in self.providers.items():
                provider.load_state_dict(previous[name])
            raise
        self.rng.setstate(state["rng"])
        self.draws, self.pending = state["draws"], copy.deepcopy(state["pending"])
        for name in ("trained", "targets", "terminal", "proposed", "replay_proposed", "observations", "tokens"):
            setattr(self, name, Counter(state[name]))
        self.committed_update, self.last_commit = state["committed_update"], state["last_commit"]

    def selection_state_dict(self):
        # The caller holds the selection/feedback lock through delivery. Selection
        # only appends pending IDs >= draws; existing entries cannot change here.
        # Save cursors, not outstanding prompts (which can contain large test suites).
        state = {p: getattr(v, "selection_state_dict", v.state_dict)() for p, v in self.providers.items()}
        return {
            "providers": state,
            "rng": self.rng.getstate(),
            "draws": self.draws,
            "proposed": dict(self.proposed),
            "replay_proposed": dict(self.replay_proposed),
        }

    def load_selection_state_dict(self, state):
        for p, v in self.providers.items():
            getattr(v, "load_selection_state_dict", v.load_state_dict)(state["providers"][p])
        self.rng.setstate(state["rng"])
        for selection in list(self.pending):
            if selection >= state["draws"]:
                del self.pending[selection]
        self.draws = state["draws"]
        self.proposed = Counter(state["proposed"])
        self.replay_proposed = Counter(state["replay_proposed"])

    def metrics(self):
        pending = Counter(entry["pool"] for entry in self.pending.values())
        result = {
            f"rollout/pool/{p}/{metric}": counter[p]
            for metric, counter in [
                ("trained", self.trained),
                ("proposed", self.proposed),
                ("terminal", self.terminal),
                ("pending", pending),
            ]
            for p in self.providers
        }
        if self.sampling["policy"] == "exposure":
            result.update({f"rollout/pool/{p}/target": self.targets[p] for p in self.providers})
        result.update({f"rollout/pool/{k}": v for k, v in self.tokens.items()})
        proposed, trained = sum(self.proposed.values()), sum(self.trained.values())
        attempted = sum(self.observations[f"{p}/attempted"] for p in self.providers)
        generated = sum(self.tokens[f"{p}/generated_tokens"] for p in self.providers)
        for name, provider in self.providers.items():
            base = getattr(provider, "base", provider)
            prefix = f"rollout/pool/{name}/"
            result.update({prefix + k: self.observations[f"{name}/{k}"] for k in pool_metrics.OBSERVATIONS})
            result.update(
                {
                    prefix + "available_rows": len(base.samples),
                    prefix + "fresh_proposed": self.proposed[name] - self.replay_proposed[name],
                    prefix + "replay_proposed": self.replay_proposed[name],
                    # Fresh rows are drawn once, so their first observations measure
                    # coverage without a growing ID set. This is not content deduplication.
                    prefix + "coverage": self.observations[f"{name}/fresh_attempted"] / max(1, len(base.samples)),
                    prefix + "proposed_share": self.proposed[name] / max(1, proposed),
                    prefix + "attempted_share": self.observations[f"{name}/attempted"] / max(1, attempted),
                    prefix + "trained_share": self.trained[name] / max(1, trained),
                    prefix + "generated_token_share": self.tokens[f"{name}/generated_tokens"] / max(1, generated),
                }
            )
            for field in (
                "generated_tokens",
                "generated_tokens_unknown_groups",
                "trained_tokens",
                "trained_tokens_unknown_groups",
                "fresh_groups",
                "replay_groups",
            ):
                result[prefix + field] = self.tokens[f"{name}/{field}"]
            if isinstance(provider, replay_provider.ReplayPromptProvider):
                result.update(
                    {
                        f"rollout/pool/{name}/replay/{key.removeprefix('rollout/prompt_replay/')}": value
                        for key, value in provider.metrics().items()
                    }
                )
        if self.catalog is not None:
            result.update({f"rollout/prompt_storage/{key}": value for key, value in self.catalog.metrics().items()})
        return result
