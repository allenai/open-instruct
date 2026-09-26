"""The FP32 diagnostic must fail rather than silently retaining BF16 tensors."""

import pytest
import torch
from scripts.miles import fp32_reference


def test_audit_rejects_low_precision_parameter():
    with pytest.raises(ValueError, match="parameters"):
        fp32_reference.audit_model(torch.nn.Linear(2, 2, dtype=torch.bfloat16))


def test_audit_rejects_low_precision_hidden_state():
    model = torch.nn.Identity()
    fp32_reference.audit_model(model)
    assert model(torch.ones(2)).dtype == torch.float32
    with pytest.raises(ValueError, match="Low-precision tensor"):
        model(torch.ones(2, dtype=torch.bfloat16))
