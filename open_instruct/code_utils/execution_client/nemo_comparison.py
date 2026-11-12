# Client-side code ported from code-execution-service revision 103ca279b35e8e81d118947f82b72c22676c6540.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Adapted comparison-only logic from NVIDIA-NeMo/Gym, commit
# bd6f5f0d8f3a56cbf3f7e1d2d4a1d2220b7ff6fe, code_gen/lcb_integration/testing_util.py.
# Modified: use scalar standard-library arithmetic instead of NumPy allclose;
# keep comparison separate from execution, timers, and early-stop behavior.
"""NeMo call-result comparison; emitted into assertion tests by the dataset helper."""

import json
import math


def _nemo_values_close(expected, predicted):
    if expected == predicted:
        return True
    if isinstance(expected, bool) or isinstance(predicted, bool):
        return False
    if not (isinstance(expected, float) or isinstance(predicted, float)):
        return False
    if not isinstance(expected, (int, float, str)) or not isinstance(predicted, (int, float, str)):
        return False
    try:
        expected, predicted = float(expected), float(predicted)
        return expected == predicted or (
            math.isfinite(expected)
            and math.isfinite(predicted)
            and abs(predicted - expected) <= 1e-8 + 1e-5 * abs(expected)
        )
    except (TypeError, ValueError):
        return False


def _nemo_call_outputs_match(expected, predicted):
    if isinstance(predicted, tuple):
        predicted = list(predicted)
    if predicted == expected:
        return True
    try:
        if json.dumps(predicted) == json.dumps(expected):
            return True
    except (TypeError, ValueError):
        pass
    if _nemo_values_close(expected, predicted):
        return True
    return (
        isinstance(predicted, list)
        and isinstance(expected, list)
        and len(predicted) == len(expected)
        and all(_nemo_values_close(want, got) for want, got in zip(expected, predicted, strict=True))
    )
