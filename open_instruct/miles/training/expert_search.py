"""Bounded sample swaps with exact, incrementally updated Core packing.

Search sees all expert replicas. Counts are a work proxy, not a runtime model.
The caller owns the final nonregression guard and retains the best admissible order.
"""

import time

import numpy as np

from open_instruct.miles.training import packing


class Partition:
    """Cached column plans and [group, pack, layer, destination] assignment counts."""

    def __init__(self, order, lengths, histograms, *, world, ep_degree, max_tokens):
        if world < 1 or not 1 < ep_degree < world or world % ep_degree or len(lengths) % world:
            raise ValueError("Search requires complete blocks and world > EP > 1")
        if not len(lengths) or sorted(order) != list(range(len(lengths))):
            raise ValueError("Search requires a complete sample permutation")
        if histograms.ndim != 3 or histograms.shape[0] != len(lengths) or histograms.shape[2] != ep_degree:
            raise ValueError("Expected one [layer, EP slot] histogram per sample")
        self.order = np.asarray(order, dtype=np.int64)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.histograms = histograms
        self.world, self.ep_degree, self.max_tokens = world, ep_degree, max_tokens
        self.columns = [self.order[r::world] for r in range(world)]
        self.plans = [packing.plan(self.lengths[c].tolist(), max_tokens) for c in self.columns]
        self.count = max(map(len, self.plans))
        self.membership = [packing.equalize(p, self.count) for p in self.plans]
        self.attention = np.stack([self._attention(r) for r in range(world)])
        self.local = np.stack([self._loads(r) for r in range(world)])
        self.loads = self.local.reshape(world // ep_degree, ep_degree, *self.local.shape[1:]).sum(axis=1)

    def _loads(self, rank):
        return np.stack([self.histograms[self.columns[rank][p]].sum(axis=0) for p in self.membership[rank]])

    def _attention(self, rank):
        lengths = self.lengths[self.columns[rank]]
        return np.array([[lengths[p].sum(), (lengths[p] ** 2).sum()] for p in self.membership[rank]])

    def work(self):
        """Three separate stage-work proxies; never pretend their units are seconds."""
        return np.array([self.loads.max(axis=(0, 3)).sum(), *self.attention.max(axis=0).sum(axis=0)], dtype=np.float64)

    def swap(self, first, second):
        """Return a candidate without mutating self; re-equalize globally if required."""
        candidate = object.__new__(Partition)
        candidate.__dict__ = self.__dict__.copy()
        candidate.order = self.order.copy()
        candidate.order[first], candidate.order[second] = self.order[second], self.order[first]
        candidate.columns = self.columns.copy()
        candidate.plans = self.plans.copy()
        touched = sorted({first % self.world, second % self.world})
        for rank in touched:
            candidate.columns[rank] = candidate.order[rank :: self.world]
            candidate.plans[rank] = packing.plan(self.lengths[candidate.columns[rank]].tolist(), self.max_tokens)
        candidate.count = max(map(len, candidate.plans))
        if candidate.count != self.count:
            # A changed world-wide pack count can alter *every* rank's equalization.
            candidate.membership = [packing.equalize(p, candidate.count) for p in candidate.plans]
            candidate.attention = np.stack([candidate._attention(r) for r in range(self.world)])
            candidate.local = np.stack([candidate._loads(r) for r in range(self.world)])
            candidate.loads = candidate.local.reshape(
                self.world // self.ep_degree, self.ep_degree, *candidate.local.shape[1:]
            ).sum(axis=1)
            return candidate, "global"
        candidate.membership = self.membership.copy()
        candidate.attention = self.attention.copy()
        candidate.local = self.local.copy()
        candidate.loads = self.loads.copy()
        unchanged = True
        for rank in touched:
            candidate.membership[rank] = packing.equalize(candidate.plans[rank], self.count)
            unchanged &= candidate.membership[rank] == self.membership[rank]
            candidate.attention[rank] = candidate._attention(rank)
        if unchanged:
            # Exact equality of position membership, not a length-band assumption.
            for position, replacement in ((first, self.order[second]), (second, self.order[first])):
                rank, row = position % self.world, position // self.world
                pack = next(i for i, members in enumerate(self.membership[rank]) if row in members)
                delta = self.histograms[replacement] - self.histograms[self.order[position]]
                candidate.local[rank, pack] += delta
                candidate.loads[rank // self.ep_degree, pack] += delta
            return candidate, "fixed"
        for rank in touched:
            candidate.local[rank] = candidate._loads(rank)
            candidate.loads[rank // self.ep_degree] += candidate.local[rank] - self.local[rank]
        return candidate, "columns"

    def positions(self):
        """Map global-list positions to their executed EP group and pack index."""
        groups = np.arange(len(self.order)) % self.world // self.ep_degree
        packs = np.empty(len(self.order), dtype=np.int64)
        for rank, membership in enumerate(self.membership):
            for pack, rows in enumerate(membership):
                packs[np.asarray(rows) * self.world + rank] = pack
        return groups, packs


def cost(loads, temperature):
    """Exact global critical work, then stable LSE for equal-work plateaus."""
    values = loads.transpose(1, 2, 0, 3).reshape(loads.shape[1], loads.shape[2], -1)
    peak = values.max(axis=-1)
    smooth = peak + temperature * np.log(np.exp((values - peak[..., None]) / temperature).sum(axis=-1))
    return int(peak.sum()), float(smooth.sum())


def objective(state, normalizers):
    """Equal-weight, arrival-normalized work proxies, with LSE only as tie-breaker.

    Unknown hardware coefficients make this a search heuristic. The caller's
    guard separately forbids regressions in either attention proxy or expert work.
    """
    exact = float((state.work() / normalizers).sum())
    smooth = cost(state.loads / normalizers[0], 0.1 / state.count)[1]
    values = state.attention / normalizers[1:]
    peaks = values.max(axis=0)
    temperature = 0.1 / state.count
    smooth += float((peaks + temperature * np.log(np.exp((values - peaks) / temperature).sum(axis=0))).sum())
    return exact, smooth


def proposals(state, rng, count=32):
    """Heavy/light complementary candidates, mixed with unrestricted random swaps."""
    groups, packs = state.positions()
    loads = state.loads
    # Rotate among near-bottlenecks, rather than repeatedly hitting one max plateau.
    excess = loads - loads.mean(axis=(0, 3), keepdims=True)
    hot = np.argsort(-excess.reshape(-1), kind="stable")[: min(8, excess.size)]
    pairs = []
    for attempt in range(count):
        if attempt % 4 == 3:
            a, b = rng.choice(len(state.order), 2, replace=False)
        else:
            group, pack, layer, slot = np.unravel_index(hot[rng.integers(len(hot))], loads.shape)
            donors = np.flatnonzero((groups == group) & (packs == pack))
            contribution = state.histograms[state.order[donors], layer, slot]
            heavy = donors[np.argsort(-contribution, kind="stable")[:4]]
            a = int(heavy[rng.integers(len(heavy))])
            others = np.flatnonzero((groups != group) | (packs != pack))
            if not len(others):
                continue
            # Cheap linearized squared-load delta considers *both* destinations,
            # all layers and slots. Exact packing/cost decides acceptance later.
            delta = state.histograms[state.order[others]] - state.histograms[state.order[a]]
            pressure = loads[group, pack] - loads[groups[others], packs[others]]
            estimate = (delta * pressure).sum(axis=(1, 2))
            if attempt % 2 == 0:
                distance = np.abs(state.lengths[state.order[others]] - state.lengths[state.order[a]])
                shortlist = np.argsort(distance, kind="stable")[: max(8, len(others) // 4)]
                others, estimate = others[shortlist], estimate[shortlist]
            light = others[np.argsort(estimate, kind="stable")[:8]]
            b = int(light[rng.integers(len(light))])
        pair = tuple(sorted((int(a), int(b))))
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def improve(state, *, seed, max_proposals, deadline, consider, statistics):
    """Strict work descent; LSE permits progress on plateaus of the exact maximum."""
    rng = np.random.default_rng(seed)
    normalizers = np.maximum(state.work(), 1)
    current = objective(state, normalizers)
    # Fixed slot ownership: copies can share each slot's demand across replicas,
    # but cannot move that demand to another slot.
    lower = int(np.ceil(state.histograms.sum(axis=0).max(axis=1) / (state.world // state.ep_degree)).sum())
    statistics.update(iterations=0, proposals=0, accepted=0, fixed=0, columns=0, global_repack=0, lower_bound=lower)
    seen, pending, stale = set(), [], 0
    for iteration in range(max_proposals):
        if time.perf_counter() >= deadline:
            statistics["stop"] = "deadline"
            break
        if (
            state.work()
            <= [lower, np.ceil(state.lengths.sum() / state.world), np.ceil((state.lengths**2).sum() / state.world)]
        ).all():
            statistics["stop"] = "lower_bound"
            break
        if stale >= 256:
            statistics["stop"] = "stagnation"
            break
        statistics["iterations"] = iteration + 1
        if not pending:
            pending = proposals(state, rng)
        if not pending:
            statistics["stop"] = "no_candidates"
            break
        pair = pending.pop()
        stale += 1
        if pair in seen:
            continue
        seen.add(pair)
        candidate, mode = state.swap(*pair)
        statistics["proposals"] += 1
        statistics["global_repack" if mode == "global" else mode] += 1
        # Every proposal can be the best guard-admissible result, even when the
        # unconstrained search has moved through a guard-inadmissible state.
        consider(candidate)
        score = objective(candidate, normalizers)
        if candidate.count <= state.count and score < current:
            state, current = candidate, score
            statistics["accepted"] += 1
            seen, pending, stale = set(), [], 0
    else:
        statistics["stop"] = "proposal_budget"
    statistics["search_objective"] = current[0]
    statistics["search_work"] = state.work().tolist()
    statistics["pack_count_lower_bound"] = int(np.ceil(state.lengths.sum() / (state.world * state.max_tokens)))
