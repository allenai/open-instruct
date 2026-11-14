"""Fixed-workload probes must reject truncation and preserve completed work."""

import importlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/miles/benchmark_fixed_tokens.py"


@pytest.mark.parametrize("truncated", [False, True])
def test_fixed_lengths_and_resume(tmp_path, monkeypatch, truncated):
    options = tmp_path / "options.json"
    options.write_text(json.dumps({"model_path": "approved-model", "context_length": 18432}))
    output = tmp_path / "output"
    calls = []

    class Engine:
        def __init__(self, **kwargs):
            assert kwargs["context_length"] == 18434

        def get_server_info(self):
            return {}

        def flush_cache(self):
            pass

        def shutdown(self):
            pass

        def generate(self, input_ids, sampling_params, stream=False):
            if not stream:
                return []
            n = sampling_params["max_new_tokens"]
            calls.append((len(input_ids[0]), n, len(input_ids)))
            return iter(
                [[{"index": i, "meta_info": {"completion_tokens": n - int(truncated)}} for i in range(len(input_ids))]]
            )

    original_import = importlib.import_module

    def module(name, *args, **kwargs):
        if name == "olmo_sglang":
            return SimpleNamespace(register=lambda: None)
        if name == "sglang":
            return SimpleNamespace(Engine=Engine)
        if name == "transformers":
            tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1, 2, 3])
            return SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: tokenizer))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", module)
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--options", str(options), "--precision", "bf16-control", "--output", str(output)]
    )
    if truncated:
        with pytest.raises(ValueError, match="Incomplete fixed-length"):
            runpy.run_path(str(SCRIPT), run_name="__main__")
        assert not (output / "complete.json").exists()
        assert not (output / "measurements.jsonl").read_text()
    else:
        runpy.run_path(str(SCRIPT), run_name="__main__")
        data = (output / "measurements.jsonl").read_bytes()
        rows = [json.loads(line) for line in data.splitlines()]
        assert len(rows) == len(calls) == 54
        # Simulate preemption after the first 45 valid cases.
        first45 = b"\n".join(data.splitlines()[:45]) + b"\n"
        (output / "measurements.jsonl").write_bytes(first45)
        (output / "complete.json").unlink()
        calls.clear()
        runpy.run_path(str(SCRIPT), run_name="__main__")
        resumed = (output / "measurements.jsonl").read_bytes()
        assert resumed.startswith(first45)
        assert len(calls) == 9
        assert all(p == 2048 and o == 16384 for p, o, _ in calls)
        assert len(resumed.splitlines()) == 54
        calls.clear()
        runpy.run_path(str(SCRIPT), run_name="__main__")
        assert calls == []
        assert (output / "measurements.jsonl").read_bytes() == resumed
