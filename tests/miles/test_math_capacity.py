"""Capacity admission must cover long requests and retain a measured memory reserve."""

import copy
from types import SimpleNamespace

import pytest
from scripts.miles import benchmark_math_capacity as pilot


def test_full_length_pool_and_graph_coverage_without_precision_changes():
    original = {"quantization": "nvfp4_online", "kv_cache_dtype": "bf16", "mem_fraction_static": 0.7}
    for concurrency in (16, 32, 64, 128):
        options = pilot.capacity_options(original, concurrency)
        assert options["max_total_tokens"] >= concurrency * (2048 + 16384 + 2)
        assert options["max_mamba_cache_size"] >= concurrency
        assert options["cuda_graph_max_bs_decode"] >= concurrency
        assert all(options[key] == value for key, value in original.items())
    with pytest.raises(ValueError):
        pilot.capacity_options(original, 256)


def test_memory_guard_uses_physical_pool_bytes_and_transient_peak():
    state = {
        "internal_states": [
            {
                "memory_usage": {"token_capacity": 393216},
                "capacity_pilot": {
                    "full_kv_bytes": 2 * pilot.GIB,
                    "peak_allocated_bytes": 24 * pilot.GIB,
                    "allocated_bytes": 20 * pilot.GIB,
                    "free_bytes": 100 * pilot.GIB,
                    "total_bytes": 288 * pilot.GIB,
                },
            }
        ]
    }
    assert pilot.next_capacity(state, {}, 32)["allowed"]
    low = copy.deepcopy(state)
    low["internal_states"][0]["capacity_pilot"]["free_bytes"] = 40 * pilot.GIB
    assert not pilot.next_capacity(low, {}, 32)["allowed"]
    del low["internal_states"][0]["capacity_pilot"]
    with pytest.raises(KeyError):
        pilot.next_capacity(low, {}, 32)


def test_streaming_counts_final_latency_and_retractions():
    class Engine:
        def generate(self, **kwargs):
            assert kwargs["sampling_params"]["max_new_tokens"] == 16384
            for count in (1, 16384):
                yield [{"index": i, "meta_info": {"completion_tokens": count, "num_retractions": i}} for i in range(2)]

    row = pilot.generate(Engine(), [1, 2], 2)
    assert row["completion_counts"] == {0: 16384, 1: 16384}
    assert row["num_retractions"] == 1
    assert set(row["request_latency_seconds"]) == {0, 1}
    assert row["output_tokens_per_second"] > 0


def test_incomplete_stream_is_not_a_success():
    class Engine:
        def generate(self, **kwargs):
            yield {"index": 0, "meta_info": {"completion_tokens": 8192}}

    with pytest.raises(ValueError, match="Incomplete"):
        pilot.generate(Engine(), [1], 1)


def test_scheduler_telemetry_preserves_native_response_envelope(monkeypatch):
    class Tensor:
        dtype, shape = "torch.uint8", [32]

        def numel(self):
            return 32

        def element_size(self):
            return 1

    class Scheduler:
        def get_internal_state(self, request):
            return SimpleNamespace(internal_state={"existing": 17})

    pool = SimpleNamespace(
        full_kv_pool=SimpleNamespace(k_buffer=[Tensor()], v_buffer=[Tensor()], mem_usage=64 / pilot.GIB),
        mamba_pool=SimpleNamespace(mamba_cache=SimpleNamespace(temporal=Tensor(), conv=[Tensor()])),
    )
    instance = Scheduler()
    instance.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(token_to_kv_pool=pool, attn_backend=SimpleNamespace())
    )
    resets = []
    cuda = SimpleNamespace(
        mem_get_info=lambda: (1000, 2000),
        memory_allocated=lambda: 500,
        memory_reserved=lambda: 600,
        max_memory_allocated=lambda: 700,
        max_memory_reserved=lambda: 800,
        reset_peak_memory_stats=lambda: resets.append(True),
    )
    scheduler = SimpleNamespace(Scheduler=Scheduler, run_scheduler_process=lambda: instance.get_internal_state(None))
    modules = {"torch": SimpleNamespace(cuda=cuda), "sglang.srt.managers.scheduler": scheduler}
    monkeypatch.setattr(pilot.importlib, "import_module", modules.__getitem__)
    response = pilot.audited_scheduler()
    assert response.internal_state["existing"] == 17
    assert response.internal_state["capacity_pilot"]["full_kv_bytes"] == 64
    assert resets == [True]
