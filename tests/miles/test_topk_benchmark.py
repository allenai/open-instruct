import base64

import numpy as np
import pytest
from scripts.miles import topk_benchmark as bench


@pytest.mark.parametrize("k", [32, 64, 128])
def test_wire_formats_preserve_candidate_arrays(k):
    values = -np.arange(3 * k, dtype=np.float32).reshape(3, k) / 10
    indices = np.arange(3 * k, dtype=np.int32).reshape(3, k)
    nested = {
        "output_top_logprobs": [
            [[float(v), int(i), None] for v, i in zip(vs, ids)] for vs, ids in zip(values, indices)
        ]
    }
    flat = {
        "output_top_logprobs_shape": [3, k],
        "output_top_logprobs_null_prefix": 0,
        "output_top_logprobs_val_flat_b64_dtype": "float32",
        "output_top_logprobs_idx_flat_b64_dtype": "int32",
        "output_top_logprobs_val_flat_b64": base64.b64encode(values.astype("<f4").tobytes()).decode(),
        "output_top_logprobs_idx_flat_b64": base64.b64encode(indices.astype("<i4").tobytes()).decode(),
    }
    for decoded in [bench.unpack(nested, k, False, 3), bench.unpack(flat, k, True, 3)]:
        np.testing.assert_array_equal(decoded[0], values)
        np.testing.assert_array_equal(decoded[1], indices)
    flat["output_top_logprobs_shape"] = [3, k - 1]
    with pytest.raises(AssertionError):
        bench.unpack(flat, k, True, 3)


def test_request_changes_extraction_without_changing_sampling():
    requests = [bench.payload([1, 2, 3], mode, 17, 256) for mode in bench.MODES]
    assert all(r["sampling_params"] == requests[0]["sampling_params"] for r in requests)
    assert all(r["logprob_start_len"] == -1 for r in requests)
    assert requests[0]["sampling_params"]["top_k"] == -1
    assert not requests[0]["return_logprob"]
    assert requests[1]["return_logprob"] and requests[1]["top_logprobs_num"] == 0


def test_latency_fractions_do_not_double_count_nested_timings():
    rows, blocks = [], []
    for mode, _, _ in bench.MODES:
        rows.append(
            dict(
                mode=mode,
                total_s=10.0,
                completion_tokens=256,
                response_bytes=100,
                client_request_pack_s=0.1,
                server_e2e_s=8.0,
                server_other_s=7.0,
                server_pack_s=1.0,
                server_json_s=0.5,
                http_residual_s=0.4,
                body_read_s=0.2,
                json_parse_s=0.6,
                array_unpack_s=0.4,
            )
        )
        blocks.append(dict(mode=mode, seconds=10.0))
    result = bench.summarize(rows, blocks)
    assert sum(result["flat_128"]["latency_fraction"].values()) == pytest.approx(1.0)
    assert result["flat_128"]["mean_latency_overhead_pct"] == 0
    assert result["flat_128"]["tokens_per_s"] == 25.6
