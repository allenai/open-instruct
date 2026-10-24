import base64
import hashlib
import io
import json

import numpy as np
import pytest
from analyze_update_audits import load_records, summarize, vector_comparison


def test_vector_comparison_separates_direction_and_scale():
    orthogonal = vector_comparison(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert orthogonal["cosine"] == 0.0
    assert orthogonal["relative_difference"] == pytest.approx(2**0.5)
    scaled = vector_comparison(np.array([2.0, 0.0]), np.array([1.0, 0.0]))
    assert scaled["cosine"] == 1.0 and scaled["relative_difference"] == 1.0
    assert vector_comparison(np.zeros(2), np.zeros(2))["cosine"] is None


def test_verified_record_and_gradient_share(tmp_path):
    arrays = {
        "gradient_norm2/embedding": np.array(25.0),
        "gradient_norm2/other": np.array(75.0),
        "rows_before/embedding": np.zeros((1, 2)),
        "rows_after/embedding": np.array([[0.03, 0.04]]),
        "row_gradients/embedding": np.array([[3.0, 4.0]]),
    }
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    raw = stream.getvalue()
    record = {
        "step": 1,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "npz_base64": base64.b64encode(raw).decode(),
        "array_count": 5,
        "token_ids": [100264],
    }
    path = tmp_path / "pod.log"
    path.write_text("timestamp H127-UPDATE-AUDIT " + json.dumps(record) + "\n")
    result = summarize(load_records(tmp_path)[1])
    assert result["gradient_norm"] == 10.0
    row = result["selected_token_rows"]["embedding"][0]
    assert row["gradient_squared_share_of_total"] == 0.25
    assert row["update_norm"] == pytest.approx(0.05)
    record["sha256"] = "0" * 64
    path.write_text("H127-UPDATE-AUDIT " + json.dumps(record))
    with pytest.raises(AssertionError):
        load_records(tmp_path)
