import json
from types import SimpleNamespace

import pytest
import torch
import triton
from scripts.train.atomic_think import fixed_kernels
from triton.runtime import autotuner


def kernel_stub():
    pass


def test_replay_uses_original_config_and_rejects_unknown_keys(tmp_path, monkeypatch):
    config = triton.Config({"BLOCK": 64}, num_warps=4)
    other = triton.Config({"BLOCK": 32}, num_warps=4)
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "tuning_choices": [
                    {
                        "function": f"{kernel_stub.__module__}.{kernel_stub.__qualname__}",
                        "key": "(128, 'torch.bfloat16')",
                        "kwargs": config.all_kwargs(),
                        "pre_hook": None,
                    }
                ],
            }
        )
    )
    tuner = SimpleNamespace(arg_names=["x", "N"], keys=["N"], base_fn=kernel_stub, configs=[other, config], cache={})
    kernels = fixed_kernels.FixedKernels(path)
    x = torch.zeros(4, dtype=torch.bfloat16)
    assert kernels.select(tuner, (x,), {"N": 128}) is config
    assert tuner.cache[(128, "torch.bfloat16")] is config
    with pytest.raises(ValueError, match="Uncalibrated"):
        kernels.select(tuner, (x, 129), {})
    tuner.configs = [other]
    with pytest.raises(ValueError, match="no longer available"):
        kernels.select(tuner, (x, 128), {})
    tuner.configs = [other, config]

    def execute(tuner, *args, **kwargs):
        tuner.best_config = tuner.cache[(128, "torch.bfloat16")]
        return "executed"

    monkeypatch.setattr(autotuner.Autotuner, "run", execute)
    monkeypatch.setattr(kernels, "validate_runtime", lambda: None)
    with pytest.raises(RuntimeError, match="intentional"), kernels:
        assert autotuner.Autotuner.run(tuner, x, N=128) == "executed"
        raise RuntimeError("intentional")
    assert autotuner.Autotuner.run is execute


def test_capture_signatures_detect_values_dtypes_and_duplicate_documents():
    doc = {"id": "one", "routes": torch.tensor([1, 2]), "nll": 0.25}
    reference = fixed_kernels.capture_signatures({"documents": [doc]})
    assert reference == fixed_kernels.capture_signatures({"documents": [dict(doc)]})
    for change in ({"routes": torch.tensor([1, 3])}, {"routes": doc["routes"].int()}, {"nll": 0.26}):
        assert reference != fixed_kernels.capture_signatures({"documents": [{**doc, **change}]})
    with pytest.raises(ValueError, match="Duplicate"):
        fixed_kernels.capture_signatures({"documents": [doc, doc]})
