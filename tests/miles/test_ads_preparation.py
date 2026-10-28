"""Token alignment and pooling contract for frozen representations."""

import numpy as np
import pytest
from scripts.data.ads import extract


def test_pooling_and_answer_likelihood_use_all_expected_positions():
    output = {
        "meta_info": {
            "hidden_states": [[[1.0, 3.0], [3.0, 5.0], [5.0, 7.0]]],
            "input_token_logprobs": [[None, 10, None], [-2.0, 20, None], [-4.0, 30, None]],
        }
    }
    vector, nll = extract.unpack(output, [10, 20, 30], 1)
    np.testing.assert_array_equal(vector, [3.0, 5.0])
    assert nll == 3.0
    with pytest.raises(ValueError, match="alignment"):
        extract.unpack(output, [10, 30, 20], 1)
    with pytest.raises(ValueError, match="hidden states"):
        extract.unpack(output, [10, 20], 1)
