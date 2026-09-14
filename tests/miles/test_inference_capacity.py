"""The capacity sweep must count completed work and retain failed-run evidence."""

import json
import sys
from types import SimpleNamespace

import pytest
from scripts.miles import inference_capacity


@pytest.mark.parametrize("shorten", [False, True])
def test_capacity_sweep_records_exact_batches_and_always_shuts_down(monkeypatch, tmp_path, shorten):
    calls = []
    closed = []

    class Engine:
        def __init__(self, **kwargs):
            assert kwargs["max_running_requests"] == 4
            assert kwargs["enable_return_routed_experts"]

        def flush_cache(self):
            pass

        def generate(self, **kwargs):
            assert kwargs["return_routed_experts"] and kwargs["return_logprob"]
            count = len(kwargs["input_ids"])
            tokens = kwargs["sampling_params"]["max_new_tokens"]
            calls.append((count, tokens))
            return [{"meta_info": {"completion_tokens": tokens - int(shorten)}} for _ in range(count)]

        def shutdown(self):
            closed.append(True)

    monkeypatch.setattr(inference_capacity.sglang, "Engine", Engine)
    monkeypatch.setattr(inference_capacity, "register", lambda: None)
    monkeypatch.setattr(inference_capacity.torch.cuda, "get_device_name", lambda: "test GPU")
    monkeypatch.setattr(
        inference_capacity.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(encode=lambda *a, **kw: list(range(600))),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capacity",
            "--model",
            "fixture",
            "--output",
            str(tmp_path),
            "--concurrencies",
            "2",
            "4",
            "--output-tokens",
            "64",
            "--repeats",
            "2",
        ],
    )
    if shorten:
        with pytest.raises(RuntimeError, match="shortened"):
            inference_capacity.main()
    else:
        inference_capacity.main()
    report = json.loads((tmp_path / "inference-capacity.json").read_text())
    assert report["status"] == ("failed" if shorten else "complete")
    assert closed == [True]
    if not shorten:
        assert calls == [(2, 256), (2, 64), (2, 64), (4, 256), (4, 64), (4, 64)]
        assert [row["warmup"] for row in report["measurements"]] == [True, True, False] * 2
        assert report["measurements"][-1]["output_tokens"] == 256
