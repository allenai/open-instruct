"""Adaptive data scheduling over frozen clusters and difficulty ranks.

ADS-inspired async adaptation: choose distinct clusters, visit their boundary
candidates, and update only after every group in that selection round is scored.
The round may span optimizer updates. All rewards precede training filtering.
"""

import copy
import hashlib
import json
import math
import random
from collections import defaultdict


class Schedule:
    def __init__(self, rows, *, seed=17, mini_size=8, active_clusters=4, alpha=0.3, band=(0.33, 0.67)):
        if not rows or mini_size < 1 or active_clusters < 1 or not 0 < alpha <= 1:
            raise ValueError("Invalid ADS configuration")
        if not 0 <= band[0] < band[1] <= 1:
            raise ValueError("Invalid ADS success band")
        self.fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        groups = defaultdict(list)
        self.lookup = {}
        for row in rows:
            identity = row["prompt_id"]
            if identity in self.lookup or not math.isfinite(row["difficulty"]):
                raise ValueError("Duplicate prompt identity or invalid difficulty")
            groups[int(row["cluster_id"])].append((float(row["difficulty"]), identity))
            self.lookup[identity] = int(row["cluster_id"])
        self.clusters = {cid: [identity for _, identity in sorted(group)] for cid, group in sorted(groups.items())}
        if len(self.clusters) < active_clusters:
            raise ValueError("Too few clusters")
        self.positions = {identity: pos for group in self.clusters.values() for pos, identity in enumerate(group)}
        self.rng = random.Random(seed)
        self.mini_size, self.active_clusters, self.alpha, self.band = mini_size, active_clusters, alpha, band
        self.reset_rounds = math.ceil(len(rows) / (mini_size * active_clusters))
        self.observed = {}
        self.reward_groups = {"zero": 0, "one": 0, "mixed": 0}
        self.queue = []
        self.rounds = {}
        self.next_round = self.epoch = self.completed_in_epoch = self.feedback_count = 0
        self._reset()

    def _reset(self):
        self.rates = {cid: 0.5 for cid in self.clusters}
        self.probs = {cid: 1 / len(self.clusters) for cid in self.clusters}
        self.mini = {cid: list(range(min(self.mini_size, len(group)))) for cid, group in self.clusters.items()}

    def take(self):
        if not self.queue:
            available = list(self.clusters)
            chosen = []
            for _ in range(self.active_clusters):
                cid = self.rng.choices(available, weights=[self.probs[c] for c in available], k=1)[0]
                chosen.append(cid)
                available.remove(cid)
            ids = [self.clusters[c][p] for c in chosen for p in self.mini[c]]
            round_id = self.next_round
            self.next_round += 1
            self.rounds[round_id] = {"epoch": self.epoch, "ids": ids, "rewards": {}}
            self.queue = [(identity, round_id, self.epoch) for identity in ids]
        return self.queue.pop(0)

    def observe(self, identity, round_id, reward):
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("ADS requires finite binary-reward group means")
        batch = self.rounds.get(round_id)
        if batch is None or identity not in batch["ids"]:
            raise ValueError("Unregistered ADS feedback")
        if identity in batch["rewards"]:
            raise ValueError("Duplicate ADS feedback")
        batch["rewards"][identity] = reward
        self.feedback_count += 1
        self.observed[identity] = self.observed.get(identity, 0) + 1
        self.reward_groups["zero" if reward == 0 else "one" if reward == 1 else "mixed"] += 1
        if len(batch["rewards"]) != len(batch["ids"]):
            return
        del self.rounds[round_id]
        if batch["epoch"] != self.epoch:
            return  # Older async work cannot move a newly reset schedule.
        by_cluster = defaultdict(list)
        directions = defaultdict(dict)
        for identity, rho in batch["rewards"].items():
            cid, pos = self.lookup[identity], self.positions[identity]
            by_cluster[cid].append(rho)
            if pos in self.mini[cid]:
                directions[cid][pos] = -1 if rho < self.band[0] else 1 if rho > self.band[1] else 0
        for cid, rates in by_cluster.items():
            self.rates[cid] = sum(rates) / len(rates)
            self._move(cid, directions[cid])
        total = sum(self.rates.values())
        for cid in self.clusters:
            target = self.rates[cid] / total if total else 1 / len(self.clusters)
            self.probs[cid] = (1 - self.alpha) * self.probs[cid] + self.alpha * target
        self.completed_in_epoch += 1
        if self.completed_in_epoch >= self.reset_rounds:
            self.epoch += 1
            self.completed_in_epoch = 0
            self._reset()

    def _move(self, cid, directions):
        old, size = self.mini[cid], len(self.clusters[cid])
        occupied = {p for p in old if directions.get(p, 0) == 0}
        for direction, positions in [
            (-1, sorted(p for p in old if directions.get(p) == -1)),
            (1, sorted((p for p in old if directions.get(p) == 1), reverse=True)),
        ]:
            for pos in positions:
                desired = min(max(pos + direction, 0), size - 1)
                candidates = [desired]
                for radius in range(1, size):
                    candidates.extend((desired + direction * radius, desired - direction * radius))
                occupied.add(next(p for p in candidates if 0 <= p < size and p not in occupied))
        self.mini[cid] = sorted(occupied)

    def state_dict(self):
        return copy.deepcopy(
            {
                "fingerprint": self.fingerprint,
                "rng": self.rng.getstate(),
                "config": [self.mini_size, self.active_clusters, self.alpha, self.band],
                **{
                    key: getattr(self, key)
                    for key in (
                        "queue",
                        "rounds",
                        "next_round",
                        "epoch",
                        "completed_in_epoch",
                        "feedback_count",
                        "observed",
                        "reward_groups",
                        "rates",
                        "probs",
                        "mini",
                    )
                },
            }
        )

    def load_state_dict(self, state):
        if state["fingerprint"] != self.fingerprint or state["config"] != [
            self.mini_size,
            self.active_clusters,
            self.alpha,
            self.band,
        ]:
            raise ValueError("ADS resume table or configuration changed")
        self.rng.setstate(state["rng"])
        for key in (
            "queue",
            "rounds",
            "next_round",
            "epoch",
            "completed_in_epoch",
            "feedback_count",
            "observed",
            "reward_groups",
            "rates",
            "probs",
            "mini",
        ):
            setattr(self, key, copy.deepcopy(state[key]))
