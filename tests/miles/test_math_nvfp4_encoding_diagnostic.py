"""Guard state isolation and context coverage in the pinned encoder diagnostic."""

import json
from types import SimpleNamespace

import pytest
import torch
from scripts.miles import diagnose_math_nvfp4_encoding as diagnostic


@pytest.mark.parametrize("fail", [False, True])
def test_replay_restores_dtype_and_keeps_input_unchanged(monkeypatch, tmp_path, fail):
    weight = torch.ones(4, 16, dtype=torch.bfloat16)
    original_dtype = torch.get_default_dtype()
    calls = []

    class Source:
        def __enter__(self):
            return self

        def __exit__(self, *unused):
            return False

        def get_tensor(self, name):
            return weight

    def quantize(tensor):
        calls.append(torch.get_default_dtype())
        assert tensor is weight
        if fail:
            raise RuntimeError("native encoder failure")
        return tensor.to(torch.uint8), tensor.to(torch.float32), torch.tensor(1.0, dtype=torch.float32)

    monkeypatch.setattr(diagnostic.torch.cuda, "set_device", lambda gpu: None)
    monkeypatch.setattr(diagnostic, "safe_open", lambda *a, **k: Source())
    monkeypatch.setattr(
        diagnostic.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(register=lambda: None)
            if name == "olmo_sglang"
            else SimpleNamespace(ModelOptNvFp4OnlineFusedMoEMethod=SimpleNamespace(_quantize_weight_nvfp4=quantize))
        ),
    )
    args = SimpleNamespace(gpu=1, model=tmp_path, output=tmp_path)
    if fail:
        with pytest.raises(RuntimeError, match="native encoder failure"):
            diagnostic.replay(args)
        assert not (tmp_path / "gpu1-complete.json").exists()
    else:
        diagnostic.replay(args)
        rows = json.loads((tmp_path / "gpu1-progress.json").read_text())
        assert len(rows) == 48 and len({row["name"] for row in rows}) == 12
        assert calls == [torch.bfloat16, torch.float32, torch.bfloat16, torch.float32] * 12
        assert json.loads((tmp_path / "gpu1-complete.json").read_text())["diagnostic_only"]
    assert torch.get_default_dtype() == original_dtype
    assert torch.equal(weight, torch.ones_like(weight))
