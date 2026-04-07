from __future__ import annotations

import threading
import unittest

import numpy as np
from parameterized import parameterized

from open_instruct import data_types
from open_instruct.replay_buffer import Fifo, ItemMetadata, Lifo, MinSize, Prioritized, ProcessedResult, Table, Uniform


def _make_result(index: int = 0) -> ProcessedResult:
    gen_result = data_types.GenerationResult(
        responses=[[1, 2, 3]],
        finish_reasons=["stop"],
        masks=[[1, 1, 1]],
        request_info=data_types.RequestInfo(
            num_calls=[0], timeouts=[0], tool_errors=[""], tool_outputs=[""], tool_runtimes=[0.0], tool_calleds=[False]
        ),
        index=index,
        prompt_id=f"prompt_{index}",
        token_statistics=data_types.TokenStatistics(
            num_prompt_tokens=10, num_response_tokens=3, generation_time=1.0, earliest_start_time=0.0
        ),
        start_time=0.0,
        logprobs=[[0.1, 0.2, 0.3]],
        reward_scores=[0.5],
        reward_metrics={},
    )
    return ProcessedResult(
        result=gen_result,
        queries=[[10, 20]],
        ground_truths=[["answer"]],
        datasets=["test_ds"],
        raw_queries=["What is 1+1?"],
        active_tools=[None],
        decoded_responses=["response text"],
        reward_scores=[0.5],
        reward_metrics={},
        percent_solved=0.5,
    )


class TestSelectors(unittest.TestCase):
    def _make_metadata(self, n: int) -> dict[str, ItemMetadata]:
        return {f"k{i}": ItemMetadata(insert_order=i, sample_count=0, priority=float(i + 1)) for i in range(n)}

    def test_uniform_selects_n(self):
        keys = ["k0", "k1", "k2", "k3"]
        meta = self._make_metadata(4)
        selected = Uniform().select(keys, meta, 2)
        assert len(selected) == 2
        assert len(set(selected)) == 2
        for k in selected:
            assert k in keys

    def test_uniform_selects_all_when_n_equals_len(self):
        keys = ["k0", "k1", "k2"]
        meta = self._make_metadata(3)
        selected = Uniform().select(keys, meta, 3)
        assert set(selected) == set(keys)

    def test_uniform_clamps_to_len(self):
        keys = ["k0", "k1"]
        meta = self._make_metadata(2)
        selected = Uniform().select(keys, meta, 10)
        assert set(selected) == set(keys)

    def test_fifo_selects_oldest(self):
        keys = ["k2", "k0", "k1"]
        meta = self._make_metadata(3)
        selected = Fifo().select(keys, meta, 2)
        assert selected == ["k0", "k1"]

    def test_lifo_selects_newest(self):
        keys = ["k0", "k1", "k2"]
        meta = self._make_metadata(3)
        selected = Lifo().select(keys, meta, 2)
        assert selected == ["k2", "k1"]

    def test_prioritized_respects_weights(self):
        np.random.seed(42)
        keys = ["k0", "k1"]
        meta = {
            "k0": ItemMetadata(insert_order=0, sample_count=0, priority=0.0),
            "k1": ItemMetadata(insert_order=1, sample_count=0, priority=1.0),
        }
        selected = Prioritized().select(keys, meta, 1)
        assert selected == ["k1"]

    def test_prioritized_zero_weights_fallback(self):
        keys = ["k0", "k1"]
        meta = {
            "k0": ItemMetadata(insert_order=0, sample_count=0, priority=0.0),
            "k1": ItemMetadata(insert_order=1, sample_count=0, priority=0.0),
        }
        selected = Prioritized().select(keys, meta, 1)
        assert len(selected) == 1
        assert selected[0] in keys


class TestMinSize(unittest.TestCase):
    def test_can_sample_false_when_empty(self):
        table = Table(max_size=10, sampler=Uniform(), remover=Fifo())
        limiter = MinSize(min_size=5)
        assert not limiter.can_sample(table)

    def test_can_sample_true_when_enough(self):
        table = Table(max_size=10, sampler=Uniform(), remover=Fifo())
        for i in range(5):
            table.insert(f"k{i}", _make_result(i))
        limiter = MinSize(min_size=5)
        assert limiter.can_sample(table)


class TestTable(unittest.TestCase):
    def test_insert_and_len(self):
        table = Table(max_size=5, sampler=Uniform(), remover=Fifo())
        assert len(table) == 0
        table.insert("a", _make_result(0))
        assert len(table) == 1

    def test_capacity_eviction_fifo(self):
        table = Table(max_size=3, sampler=Uniform(), remover=Fifo())
        for i in range(5):
            table.insert(f"k{i}", _make_result(i))
        assert len(table) == 3
        with table._can_sample:
            assert "k0" not in table._data
            assert "k1" not in table._data
            assert "k2" in table._data

    def test_capacity_eviction_lifo(self):
        table = Table(max_size=3, sampler=Uniform(), remover=Lifo())
        for i in range(5):
            table.insert(f"k{i}", _make_result(i))
        assert len(table) == 3
        with table._can_sample:
            assert "k0" in table._data
            assert "k1" in table._data
            assert "k2" in table._data
            assert "k3" not in table._data
            assert "k4" not in table._data

    def test_sample_returns_n_items(self):
        table = Table(max_size=5, sampler=Uniform(), remover=Fifo(), max_times_sampled=10)
        for i in range(5):
            table.insert(f"k{i}", _make_result(i))
        results = table.sample(3)
        assert results is not None
        assert len(results) == 3

    def test_sample_evicts_after_max_times_sampled(self):
        table = Table(max_size=5, sampler=Uniform(), remover=Fifo(), max_times_sampled=1, rate_limiter=MinSize(3))
        for i in range(3):
            table.insert(f"k{i}", _make_result(i))
        results = table.sample(3)
        assert results is not None
        assert len(results) == 3
        assert len(table) == 0

    def test_fifo_equivalent_default(self):
        """Default config: insert N, sample N, all consumed, buffer empty."""
        n = 4
        table = Table(max_size=n, sampler=Uniform(), remover=Fifo(), max_times_sampled=1, rate_limiter=MinSize(n))
        inserted = []
        for i in range(n):
            r = _make_result(i)
            table.insert(f"k{i}", r)
            inserted.append(r)
        results = table.sample(n)
        assert results is not None
        assert len(results) == n
        result_prompts = {r.result.prompt_id for r in results}
        inserted_prompts = {r.result.prompt_id for r in inserted}
        assert result_prompts == inserted_prompts
        assert len(table) == 0

    def test_shutdown_unblocks_sample(self):
        table = Table(max_size=10, sampler=Uniform(), remover=Fifo(), rate_limiter=MinSize(5))
        result_holder = [None]

        def sample_thread():
            result_holder[0] = table.sample(5)

        t = threading.Thread(target=sample_thread)
        t.start()
        table.shutdown()
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert result_holder[0] is None

    def test_sample_blocks_until_min_size(self):
        table = Table(max_size=10, sampler=Uniform(), remover=Fifo(), max_times_sampled=1, rate_limiter=MinSize(3))
        result_holder = [None]
        event = threading.Event()

        def sample_thread():
            event.set()
            result_holder[0] = table.sample(3)

        t = threading.Thread(target=sample_thread)
        t.start()
        event.wait()
        assert result_holder[0] is None
        for i in range(3):
            table.insert(f"k{i}", _make_result(i))
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert result_holder[0] is not None
        assert len(result_holder[0]) == 3

    def test_concurrent_insert_and_sample(self):
        n = 20
        table = Table(max_size=n * 2, sampler=Uniform(), remover=Fifo(), max_times_sampled=1, rate_limiter=MinSize(n))
        all_results = []
        lock = threading.Lock()

        def inserter():
            for i in range(n):
                table.insert(f"k{i}", _make_result(i))

        def sampler():
            results = table.sample(n)
            with lock:
                all_results.extend(results or [])

        t_insert = threading.Thread(target=inserter)
        t_sample = threading.Thread(target=sampler)
        t_insert.start()
        t_sample.start()
        t_insert.join(timeout=5.0)
        t_sample.join(timeout=5.0)
        assert not t_insert.is_alive()
        assert not t_sample.is_alive()
        assert len(all_results) == n

    @parameterized.expand([("max_times_2",), ("max_times_3",)])
    def test_max_times_sampled_parametrized(self, name):
        max_times = int(name.split("_")[-1])
        table = Table(
            max_size=5, sampler=Uniform(), remover=Fifo(), max_times_sampled=max_times, rate_limiter=MinSize(1)
        )
        table.insert("k0", _make_result(0))
        for _ in range(max_times - 1):
            results = table.sample(1)
            assert results is not None
            assert len(table) == 1
        results = table.sample(1)
        assert results is not None
        assert len(table) == 0


if __name__ == "__main__":
    unittest.main()
