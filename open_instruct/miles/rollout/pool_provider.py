"""Finite named pools with trained-exposure control and pool-local fresh replay."""

import copy
import math
import random
from collections import Counter

from open_instruct.miles.rollout import replay_provider

SELECTION_ID = "pool_selection_id"
POOL = "sampling_pool"


class FinitePrompts:
    requires_feedback = False

    def __init__(self, samples, *, seed, shuffle):
        self.samples = list(samples)
        if shuffle:
            random.Random(seed).shuffle(self.samples)
        self.offset = 0

    def available(self):
        return self.offset < len(self.samples)

    def next_samples(self, n):
        if n > len(self.samples) - self.offset:
            raise StopIteration("Fresh pool exhausted")
        result = self.samples[self.offset : self.offset + n]
        self.offset += n
        return result

    def state_dict(self):
        return {"offset": self.offset}

    def load_state_dict(self, state):
        offset = state["offset"]
        if type(offset) is not int or not 0 <= offset <= len(self.samples):
            raise ValueError("Invalid finite pool cursor")
        self.offset = offset


class ExposurePromptProvider:
    requires_feedback = True

    def __init__(self, samples, *, pools, seed, shuffle, identity, replay=None):
        self.identity = copy.deepcopy(
            {"version": 1, "manifest": identity, "pools": pools, "seed": seed, "shuffle": shuffle, "replay": replay}
        )
        self.rng = random.Random(seed)
        self.providers = {}
        for name in pools:
            members = [s for s in samples if s.metadata[POOL] == name]
            base = FinitePrompts(members, seed=f"{seed}:{name}", shuffle=shuffle)
            self.providers[name] = (
                replay_provider.ReplayPromptProvider(base, members, **replay) if replay and members else base
            )
        if any(s.metadata[POOL] not in pools for s in samples):
            raise ValueError("Unknown runtime pool")
        self.draws = 0
        self.pending = {}
        self.trained = Counter()
        self.targets = Counter()
        self.terminal = Counter()
        self.proposed = Counter()
        self.tokens = Counter()
        self.committed_update = -1

    def active(self):
        return [name for name, provider in self.providers.items() if provider.available()]

    def available(self):
        return bool(self.active())

    def next_samples(self, n):
        if type(n) is not int or n < 0:
            raise ValueError("n must be a nonnegative integer")
        result = []
        for _ in range(n):
            active = self.active()
            if not active:
                raise StopIteration("All named pools exhausted")
            outstanding = Counter(entry["pool"] for entry in self.pending.values())
            weights = []
            for name in active:
                delivery = max(0.05, (self.trained[name] + 1) / (self.terminal[name] + 2))
                deficit = self.targets[name] - self.trained[name] - outstanding[name] * delivery
                # Bound proposal ratios (20:1) and use a 64-group correction horizon.
                weight = math.exp(max(-3, min(3, deficit / 64))) / delivery
                weights.append(max(0.25, min(5.0, weight)))
            name = self.rng.choices(active, weights=weights, k=1)[0]
            sample = copy.copy(self.providers[name].next_samples(1)[0])
            sample.metadata = sample.metadata | {SELECTION_ID: self.draws}
            self.pending[self.draws] = {"pool": name, "active": active, "metadata": dict(sample.metadata)}
            self.draws += 1
            self.proposed[name] += 1
            result.append(sample)
        return result

    def feedback(self, records):
        for record in records:
            entry = self.pending.get(record.get(SELECTION_ID))
            if entry is None:
                continue
            if record.get("kind") == "group" and not entry.get("observed"):
                entry["observed"] = True
                lengths = [r.get("response_tokens") for r in record.get("responses", [])]
                if lengths and all(type(n) is int and n >= 0 for n in lengths):
                    entry["response_tokens"] = sum(lengths)
                    self.tokens[f"{entry['pool']}/generated_tokens"] += sum(lengths)
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
            for active in entry["active"]:
                self.targets[active] += 1 / len(entry["active"])
        self.committed_update = update
        self.last_commit = list(selections)

    def state_dict(self):
        return {
            "kind": "pool_exposure",
            "identity": copy.deepcopy(self.identity),
            "rng": self.rng.getstate(),
            "providers": {p: v.state_dict() for p, v in self.providers.items()},
            "draws": self.draws,
            "pending": copy.deepcopy(self.pending),
            "trained": dict(self.trained),
            "targets": dict(self.targets),
            "terminal": dict(self.terminal),
            "proposed": dict(self.proposed),
            "tokens": dict(self.tokens),
            "committed_update": self.committed_update,
            "last_commit": getattr(self, "last_commit", None),
        }

    def load_state_dict(self, state):
        if state.get("kind") != "pool_exposure" or state.get("identity") != self.identity:
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
        for field in ("trained", "targets", "terminal", "proposed"):
            if not set(state[field]) <= self.providers.keys() or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in state[field].values()
            ):
                raise ValueError("Invalid pool checkpoint counters")
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
        for name in ("trained", "targets", "terminal", "proposed", "tokens"):
            setattr(self, name, Counter(state[name]))
        self.committed_update, self.last_commit = state["committed_update"], state["last_commit"]

    def selection_state_dict(self):
        # Selection rollback is serialized against feedback; omit large replay histories.
        state = {p: getattr(v, "selection_state_dict", v.state_dict)() for p, v in self.providers.items()}
        return {
            "providers": state,
            "rng": self.rng.getstate(),
            "draws": self.draws,
            "pending": copy.deepcopy(self.pending),
            "proposed": dict(self.proposed),
        }

    def load_selection_state_dict(self, state):
        for p, v in self.providers.items():
            getattr(v, "load_selection_state_dict", v.load_state_dict)(state["providers"][p])
        self.rng.setstate(state["rng"])
        self.draws, self.pending = state["draws"], copy.deepcopy(state["pending"])
        self.proposed = Counter(state["proposed"])

    def metrics(self):
        pending = Counter(entry["pool"] for entry in self.pending.values())
        result = {
            f"rollout/pool/{p}/{metric}": counter[p]
            for metric, counter in [
                ("trained", self.trained),
                ("target", self.targets),
                ("proposed", self.proposed),
                ("terminal", self.terminal),
                ("pending", pending),
            ]
            for p in self.providers
        }
        result.update({f"rollout/pool/{k}": v for k, v in self.tokens.items()})
        for name, provider in self.providers.items():
            if isinstance(provider, replay_provider.ReplayPromptProvider):
                result.update(
                    {
                        f"rollout/pool/{name}/replay/{key.removeprefix('rollout/prompt_replay/')}": value
                        for key, value in provider.metrics().items()
                    }
                )
        return result
