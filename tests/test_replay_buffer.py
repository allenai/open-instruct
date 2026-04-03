import collections
import unittest

from parameterized import parameterized

from open_instruct.replay_buffer import (
    EvictionType,
    ReplayBuffer,
    ReplayGroup,
    SamplerType,
    SumTree,
)


def _make_group(group_id: str, insertion_step: int = 0, priority: float = 1.0, n: int = 4) -> ReplayGroup:
    return ReplayGroup(
        group_id=group_id,
        queries=[[1, 2, 3]] * n,
        responses=[[4, 5]] * n,
        masks=[[1, 1]] * n,
        logprobs=[[-0.5, -0.3]] * n,
        scores=[1.0] * n,
        finish_reasons=["stop"] * n,
        insertion_step=insertion_step,
        priority=priority,
    )


class TestSumTree(unittest.TestCase):
    def test_add_and_total(self):
        tree = SumTree(4)
        tree.add("a", 1.0)
        tree.add("b", 2.0)
        tree.add("c", 3.0)
        self.assertAlmostEqual(tree.total(), 6.0)
        self.assertEqual(len(tree), 3)

    def test_update(self):
        tree = SumTree(4)
        tree.add("a", 1.0)
        tree.add("b", 2.0)
        tree.update("a", 5.0)
        self.assertAlmostEqual(tree.total(), 7.0)

    def test_remove(self):
        tree = SumTree(4)
        tree.add("a", 1.0)
        tree.add("b", 2.0)
        tree.remove("a")
        self.assertAlmostEqual(tree.total(), 2.0)
        self.assertEqual(len(tree), 1)

    def test_add_duplicate_updates(self):
        tree = SumTree(4)
        tree.add("a", 1.0)
        tree.add("a", 3.0)
        self.assertAlmostEqual(tree.total(), 3.0)
        self.assertEqual(len(tree), 1)


class TestReplayBufferFIFO(unittest.TestCase):
    def test_fifo_insert_sample_order(self):
        buf = ReplayBuffer(capacity=4, sampler_type="fifo", eviction_type="fifo")
        groups = [_make_group(f"g{i}", insertion_step=i) for i in range(4)]
        buf.insert(groups)
        sampled = buf.sample(4)
        self.assertEqual([g.group_id for g in sampled], ["g0", "g1", "g2", "g3"])

    def test_fifo_recovers_use_once_behavior(self):
        buf = ReplayBuffer(capacity=2, sampler_type="fifo", eviction_type="after_n_samples", max_sample_count=1)
        buf.insert([_make_group("g0"), _make_group("g1")])
        sampled = buf.sample(2)
        self.assertEqual(len(sampled), 2)
        self.assertEqual(len(buf), 0)

    def test_fifo_eviction_on_overflow(self):
        buf = ReplayBuffer(capacity=3, sampler_type="fifo", eviction_type="fifo")
        buf.insert([_make_group(f"g{i}") for i in range(3)])
        buf.insert([_make_group("g3")])
        self.assertEqual(len(buf), 3)
        sampled = buf.sample(3)
        ids = [g.group_id for g in sampled]
        self.assertNotIn("g0", ids)
        self.assertIn("g3", ids)


class TestReplayBufferUniform(unittest.TestCase):
    def test_uniform_hits_all_items(self):
        buf = ReplayBuffer(capacity=5, sampler_type="uniform", eviction_type="fifo", seed=42)
        buf.insert([_make_group(f"g{i}") for i in range(5)])
        seen = collections.Counter()
        for _ in range(200):
            sampled = buf.sample(3)
            for g in sampled:
                seen[g.group_id] += 1
            for g in sampled:
                g.sample_count = 0
        for i in range(5):
            self.assertGreater(seen[f"g{i}"], 0, f"g{i} was never sampled")


class TestReplayBufferPrioritized(unittest.TestCase):
    def test_prioritized_proportional(self):
        buf = ReplayBuffer(capacity=3, sampler_type="prioritized", eviction_type="fifo", seed=42)
        buf.insert([
            _make_group("low", priority=1.0),
            _make_group("high", priority=100.0),
        ])
        counts = collections.Counter()
        for _ in range(500):
            sampled = buf.sample(1)
            counts[sampled[0].group_id] += 1
            sampled[0].sample_count = 0
        self.assertGreater(counts["high"], counts["low"] * 5)

    def test_update_priorities(self):
        buf = ReplayBuffer(capacity=3, sampler_type="prioritized", eviction_type="fifo", seed=42)
        buf.insert([
            _make_group("a", priority=1.0),
            _make_group("b", priority=1.0),
        ])
        buf.update_priorities({"a": 100.0})
        counts = collections.Counter()
        for _ in range(500):
            sampled = buf.sample(1)
            counts[sampled[0].group_id] += 1
            sampled[0].sample_count = 0
        self.assertGreater(counts["a"], counts["b"] * 5)


class TestEviction(unittest.TestCase):
    def test_after_n_samples_eviction(self):
        buf = ReplayBuffer(capacity=4, sampler_type="uniform", eviction_type="after_n_samples", max_sample_count=2, seed=0)
        buf.insert([_make_group("g0")])
        buf.sample(1)
        self.assertEqual(len(buf), 1)
        buf.sample(1)
        self.assertEqual(len(buf), 0)

    def test_fifo_eviction_removes_oldest(self):
        buf = ReplayBuffer(capacity=2, sampler_type="uniform", eviction_type="fifo")
        buf.insert([_make_group("g0"), _make_group("g1")])
        buf.insert([_make_group("g2")])
        self.assertEqual(len(buf), 2)
        keys = list(buf._groups.keys())
        self.assertEqual(keys, ["g1", "g2"])


class TestStateDictRoundTrip(unittest.TestCase):
    @parameterized.expand([
        ("fifo_fifo", "fifo", "fifo"),
        ("uniform_fifo", "uniform", "fifo"),
        ("prioritized_after_n", "prioritized", "after_n_samples"),
    ])
    def test_round_trip(self, _name, sampler, eviction):
        buf = ReplayBuffer(capacity=10, sampler_type=sampler, eviction_type=eviction, max_sample_count=3, seed=123)
        buf.insert([_make_group(f"g{i}", priority=float(i + 1)) for i in range(5)])
        buf.sample(2)

        state = buf.state_dict()
        buf2 = ReplayBuffer(capacity=1)
        buf2.load_state_dict(state)

        self.assertEqual(len(buf2), len(buf))
        self.assertEqual(buf2.capacity, buf.capacity)
        self.assertEqual(buf2.sampler_type, buf.sampler_type)
        self.assertEqual(buf2.eviction_type, buf.eviction_type)
        self.assertEqual(buf2.max_sample_count, buf.max_sample_count)
        self.assertEqual(list(buf2._groups.keys()), list(buf._groups.keys()))
        for key in buf._groups:
            self.assertEqual(buf2._groups[key].sample_count, buf._groups[key].sample_count)
            self.assertEqual(buf2._groups[key].priority, buf._groups[key].priority)


class TestEdgeCases(unittest.TestCase):
    def test_sample_empty_buffer_raises(self):
        buf = ReplayBuffer(capacity=4, sampler_type="fifo")
        with self.assertRaises(ValueError):
            buf.sample(1)

    def test_sample_more_than_available_raises(self):
        buf = ReplayBuffer(capacity=4, sampler_type="uniform")
        buf.insert([_make_group("g0")])
        with self.assertRaises(ValueError):
            buf.sample(2)

    def test_sample_zero_returns_empty(self):
        buf = ReplayBuffer(capacity=4, sampler_type="fifo")
        buf.insert([_make_group("g0")])
        self.assertEqual(buf.sample(0), [])

    def test_update_priority_nonexistent_key_is_noop(self):
        buf = ReplayBuffer(capacity=4, sampler_type="prioritized")
        buf.update_priorities({"nonexistent": 5.0})
        self.assertEqual(len(buf), 0)


if __name__ == "__main__":
    unittest.main()
